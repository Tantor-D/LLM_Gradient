import json
import os
from hashlib import md5
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from peft import PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import RobertaModel

# 用于输出的一些设置
has_print_size = False


def prepare_batch(batch, device=torch.device("cuda:0")):
    """ Move the batch to the device. """
    for key in batch:
        batch[key] = batch[key].to(device)


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """ 
    Retrieve the highest index for which the data (either representation or gradients) has been stored. 

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """

    files = [file for file in os.listdir(
        output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1])
             for file in files]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


def get_output(model,
               weights: Iterable[Tensor],
               buffers: Iterable[Tensor],
               input_ids=None,
               attention_mask=None,
               labels=None,
               ) -> Tensor:
    logits = model(weights, buffers, *(input_ids.unsqueeze(0),
                                       attention_mask.unsqueeze(0))).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    return loss


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    """
    CudaProjector 类主要用于将梯度或特征进行投影操作。这通常包括将高维梯度或特征映射到低维空间，从而减少计算复杂度或实现某些降维效果。
    CudaProjector 类可能提供多种投影方法，如随机投影、主成分分析（PCA）、线性判别分析（LDA）等。用户可以根据具体需求选择适合的投影方法。
    CudaProjector 可能与深度学习框架（如 PyTorch）紧密集成，方便用户在训练神经网络时进行梯度投影操作。
    """

    # 根据是否用显卡当以不同的投影器
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    """
    首先，函数检查输入的 model 是否是 PeftModel 类型。
    如果是 PeftModel 类型，它会进一步检查模型中是否存在不应该需要梯度的参数。
    
    检查非LoRA参数是否需要梯度：
    对于 PeftModel 类型的模型，函数会遍历所有参数的名称和对应的参数值，筛选出那些需要梯度且名称中不包含“lora”的参数。
    如果发现这样的参数存在，函数会通过断言（assert）引发一个错误，以确保在 PEFT 模型中，只有 LoRA 参数需要梯度。
    
    计算需要梯度的参数数量：
    无论模型是否为 PeftModel 类型，函数都会计算并返回模型中所有需要梯度的参数的数量。
    它通过遍历 model.parameters() 并对需要梯度的参数的元素数量（numel()）求和来实现这一点。
    """
    # todo 这些参数到底长什么样子要调试来看看
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                      for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params


def obtain_gradients(model, batch):
    """ obtain gradients. """
    loss = model(**batch).loss
    loss.backward()
    vectorized_grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads


def obtain_sign_gradients(model, batch):
    """ obtain gradients with sign. """
    loss = model(**batch).loss
    loss.backward()

    # Instead of concatenating the gradients, concatenate their signs
    vectorized_grad_signs = torch.cat(
        [torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])

    return vectorized_grad_signs


def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    loss = model(**batch).loss
    loss.backward()

    # model.named_parameters() 返回一个生成器，生成模型中所有参数的名称和参数值对。
    # n 是参数的名称，p 是参数的张量
    # 检查每个参数 p 是否有对应的梯度（即 p.grad 是否为 None）
    # 将每个参数 p 的梯度 p.grad 展平为一维向量。view(-1) 将张量展平为一维。
    # 最后用到了torch.cat()函数，将所有参数的梯度拼接成一个张量。
    # torch.cat 意味着除了在拼接的维度上，所有的维度都是相同的
    # 也就是说，这里把一个数据的梯度统一的拼接到了一起，成了一个shape为(x,)的向量
    vectorized_grads = torch.cat([p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

    # 由于每一层的梯度的维度都一样，因此只需要知道每一层的梯度的维度，然后知道拼接的顺序，就可以知道每一层的梯度在vectorized_grads中的位置

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads


def prepare_optimizer_state(model, optimizer_state, device):
    # import ipdb; ipdb.set_trace()
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                        for n in names])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


def collect_grads(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam",
                  max_samples: Optional[int] = None):
    """
    Collects gradients from the model during evaluation and saves them to disk.
    这个函数只会处理一个checkpoint时的梯度，所以在上层需要使用脚本多次调用这个函数来获取多个梯度。

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states. 
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 16  # batch size for the projectors
    torch.random.manual_seed(0)  # set the random seed for torch

    project_interval = 16  # project every 16 batches
    save_interval = 160  # save every 160 batches

    def _project(current_full_grads, projected_grads):
        # 将 current_full_grads 列表中的所有梯度堆叠成一个张量。
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)
        for i, projector in enumerate(projectors):
            # 遍历 projectors 列表中的每一个投影器 projector，将 current_full_grads 投影到 projector 的维度 proj_dim[i] 上。
            # note 设置映射的维度为1时，不进行映射，直接保存
            if proj_dim[i] != 1:
                current_projected_grads = projector.project(current_full_grads, model_id=model_id)
            else:
                current_projected_grads = current_full_grads
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    def _save(projected_grads, output_dirs):
        # 将 projected_grads 中的梯度保存到 output_dirs 中。
        for dim in proj_dim:
            # 没有数据的话，就不处理
            if len(projected_grads[dim]) == 0:
                continue

            # 因为 projected_grads[dim] 是一个列表，所以需要将其中的所有张量拼接成一个张量。且不会新增一个维度（stack会）
            projected_grads[dim] = torch.cat(projected_grads[dim])

            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count}.pt")

            # 对梯度进行保存，注意保存完后要从 projected_grads里删掉
            torch.save(projected_grads[dim], outfile)
            print(f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
            projected_grads[dim] = []

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    print("dtype: " + str(dtype))

    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        # 这个函数只会处理一个checkpoint时的梯度，所以在上层需要使用脚本多次调用这个函数来获取多个梯度
        # 由于是一个checkpoint，因此直接加载就好，后续也不用变，整个checkpoint计算时都是一样的
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    # 这个projector是一个构造函数，具体是用于确定是用trak'库的CudaProjector还是BasicProjector
    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model)

    # initialize a project for each target projector dimension
    projectors = []
    for dim in proj_dim:
        proj = projector(grad_dim=number_of_params,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.rademacher,
                         device=device,
                         dtype=dtype,
                         block_size=block_size,
                         max_batch_size=projector_batch_size)
        projectors.append(proj)

    count = 0

    # set up a output directory for each dimension
    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)

    # max index for each dimension，用于训练中断后跳过已经做过的
    max_index = min(get_max_saved_index(output_dirs[dim], "grads") for dim in proj_dim)

    # projected_gradients
    full_grads = []  # full gradients
    projected_grads = {dim: [] for dim in proj_dim}  # projected gradients，是一个字典，key为被映射到的维度

    # 计算样本的梯度
    for batch in tqdm(dataloader, total=len(dataloader)):
        prepare_batch(batch)  # 移到cuda上
        count += 1

        # 这里会跳过已经做过了的，用于训练中断了之后继续
        if count <= max_index:
            print("skipping count", count)
            continue

        # 计算梯度
        if gradient_type == "adam":
            if count == 1:
                print("Using Adam gradients")
            vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
        elif gradient_type == "sign":
            if count == 1:
                print("Using Sign gradients")
            vectorized_grads = obtain_sign_gradients(model, batch)
        else:
            if count == 1:
                print("Using SGD gradients")
            vectorized_grads = obtain_gradients(model, batch)

        # add the gradients to the full_grads
        full_grads.append(vectorized_grads)
        model.zero_grad()

        # 数量有一定了之后就投影一次，然后清空full_grads
        if count % project_interval == 0:
            _project(full_grads, projected_grads)
            full_grads = []

        # 数量有一定了之后就存一次
        if count % save_interval == 0:
            _save(projected_grads, output_dirs)

        # 这里可以规定需要的sample数量，训练集那里应该设置为空，测试集可以设置一个上限
        if max_samples is not None and count == max_samples:
            break

    # 没到设定数量的剩余的没映射的样本
    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []

    # 把所有维度计算出来的信息都保存
    for dim in proj_dim:
        _save(projected_grads, output_dirs)

    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        merge_and_normalize_info(output_dir, prefix="grads")
        merge_info(output_dir, prefix="grads")

    print("collect_grads() finished")


def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    # 筛选仅仅处理prefix开头的数据，因为grads和loss之类的数据都是存在一起的。
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]

    # Sort the files in ascending order
    # 排序依据是将文件名按照“-”分割后取第二部分转换为整数
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))

    # 合并和标准化数据
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def collect_reps(dataloader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 output_dir: str,
                 max_samples: Optional[int] = None):
    """
    Collects representations from a dataloader using a given model and saves them to the output directory.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.
        model (torch.nn.Module): The model used to compute the representations.
        output_dir (str): The directory where the representations will be saved.
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    all_reps = []
    count = 0
    save_interval = 160  # save every 160 batches

    device = next(model.parameters()).device  # only works for single gpu
    max_index = get_max_saved_index(output_dir, prefix="reps")

    for batch in tqdm(dataloader):
        count += 1
        if count <= max_index:
            print("skipping count", count)
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            if isinstance(model, RobertaModel):
                reps = model(input_ids=input_ids,
                             attention_mask=attention_mask, output_hidden_states=True, return_dict=True).pooler_output
            else:
                hidden_states = model(input_ids,
                                      labels=input_ids,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True).hidden_states
                ids = torch.arange(len(input_ids), device=input_ids.device)
                pos = attention_mask.sum(dim=1) - 1
                reps = hidden_states[-1][ids, pos]

            all_reps.append(reps.cpu())
            if count % save_interval == 0:
                all_reps = torch.cat(all_reps)
                outfile = os.path.join(output_dir, f"reps-{count}.pt")
                torch.save(all_reps, outfile)
                all_reps = []
                print(f"Saving {outfile}")

            if max_samples is not None and count >= max_samples:
                break

    if len(all_reps) > 0:
        all_reps = torch.cat(all_reps)
        outfile = os.path.join(output_dir, f"reps-{count}.pt")
        torch.save(all_reps, outfile)
        print(f"Saving {outfile}")

    torch.cuda.empty_cache()
    merge_and_normalize_info(output_dir, prefix="reps")

    print("Finished")


def get_loss(dataloader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             output_dir: str, ):
    """ Get the loss of the model on the given dataset. """
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(dataloader):
        prepare_batch(batch)
        num_token = (batch["labels"] != -100).sum()
        with torch.inference_mode():
            loss = model(**batch).loss * num_token
        total_loss += loss.item()
        total_tokens += num_token.item()

    print(f"Loss: {total_loss / total_tokens}")
    result = {"num_tokens": total_tokens, "loss": (
            total_loss / total_tokens)}
    with open(os.path.join(output_dir, "loss.txt"), "w") as f:
        f.write(json.dumps(result, indent=4))
