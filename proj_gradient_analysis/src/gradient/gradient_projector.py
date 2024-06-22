# class Gradient_Projector():
#
#     def __init__(self, config):
#         pass
#
#     pass
#     # 负责读取模型的梯度，然后进行投影
#     # 输入应该是训练的一些信息、梯度的信息、层的大小之类（或者层数）的
#     # 可以配置每几层进行一次投影，投影的方式是什么？
#     # 输出应该是投影后的梯度
#     # 应该可以设置
#
#
# """
#     This script is used for getting gradients or representations of a pre-trained model, a lora model, or a peft-initialized model for a given task.
# """
#
# import argparse
# import os
# import pdb
# from copy import deepcopy
# from typing import Any
#
# import torch
# from peft import LoraConfig, PeftModel, TaskType, get_peft_model
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# from less.data_selection.collect_grad_reps import (collect_grads, collect_reps,
#                                                    get_loss)
# from less.data_selection.get_training_dataset import get_training_dataset
# from less.data_selection.get_validation_dataset import (get_dataloader,
#                                                         get_dataset)
#
#
# def load_model(model_name_or_path: str,
#                torch_dtype: Any = torch.bfloat16) -> Any:
#     """
#     Load a model from a given model name or path.
#
#     Args:
#         model_name_or_path (str): The name or path of the model.
#         torch_dtype (Any, optional): The torch data type. Defaults to torch.bfloat16.
#
#     Returns:
#         Any: The loaded model.
#     """
#
#     # true，如果是走的checkpoint的话
#     is_peft = os.path.exists(os.path.join(
#         model_name_or_path, "adapter_config.json"))
#
#     if is_peft:
#         # load this way to make sure that optimizer states match the model structure
#         config = LoraConfig.from_pretrained(model_name_or_path, ignore_mismatched_sizes=True)
#         base_model = AutoModelForCausalLM.from_pretrained(
#             config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto")
#         model = PeftModel.from_pretrained(
#             base_model, model_name_or_path, device_map="auto")
#     else:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name_or_path, torch_dtype=torch_dtype, device_map="auto")
#
#     for name, param in model.named_parameters():
#         if 'lora' in name or 'Lora' in name:
#             param.requires_grad = True
#     return model
#
#
# def get_args():
#     parser = argparse.ArgumentParser(description='Script for getting validation gradients')
#     parser.add_argument('--task', type=str, default=None,
#                         help='Specify the task from bbh, tydiqa or mmlu. One of variables of task and train_file must be specified')
#     parser.add_argument("--train_file", type=str, default=None,
#                         help="The path to the training data file we'd like to obtain the gradients/representations for. One of variables of task and train_file must be specified")
#     parser.add_argument("--info_type",
#                         choices=["grads", "reps", "loss"], help="The type of information")
#     parser.add_argument("--model_path", type=str,
#                         default=None, help="The path to the model")
#     parser.add_argument("--max_samples", type=int,
#                         default=None, help="The maximum number of samples")
#     parser.add_argument("--torch_dtype", type=str, default="bfloat16",
#                         choices=["float32", "bfloat16"], help="The torch data type")
#     parser.add_argument("--output_path", type=str,
#                         default=None, help="The path to the output")
#     parser.add_argument("--data_dir", type=str,
#                         default=None, help="The path to the data")
#     parser.add_argument("--gradient_projection_dimension", nargs='+',
#                         help="The dimension of the projection, can be a list", type=int, default=[8192])
#     parser.add_argument("--gradient_type", type=str, default="adam",
#                         choices=["adam", "sign", "sgd"], help="The type of gradient")
#     parser.add_argument("--chat_format", type=str,
#                         default="tulu", help="The chat format")
#     parser.add_argument("--use_chat_format", type=bool,
#                         default=True, help="Whether to use chat format")
#     parser.add_argument("--max_length", type=int, default=2048,
#                         help="The maximum length")
#     parser.add_argument("--zh", default=False, action="store_true",
#                         help="Whether we are loading a translated chinese version of tydiqa dev data (Only applicable to tydiqa)")
#     parser.add_argument("--initialize_lora", default=False, action="store_true",
#                         help="Whether to initialize the base model with lora, only works when is_peft is False")
#     parser.add_argument("--lora_r", type=int, default=8,
#                         help="The value of lora_r hyperparameter")
#     parser.add_argument("--lora_alpha", type=float, default=32,
#                         help="The value of lora_alpha hyperparameter")
#     parser.add_argument("--lora_dropout", type=float, default=0.1,
#                         help="The value of lora_dropout hyperparameter")
#     parser.add_argument("--lora_target_modules", nargs='+',
#                         default=["q_proj", "k_proj", "v_proj", "o_proj"], help="The list of lora_target_modules")
#     args = parser.parse_args()
#     assert args.task is not None or args.train_file is not None
#     return args
#
#
#
# def main(args):
#     # 首先把模型给加载出来
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#
#     # 注意这里走的是load_model，它会根据上次跑的时候留下来的config智能的加载模型
#     dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
#     model = load_model(args.model_path, dtype)
#
#     # pad token is not added by default for pretrained models
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({"pad_token": "<pad>"})
#
#     # resize embeddings if needed (e.g. for LlamaTokenizer)
#     embedding_size = model.get_input_embeddings().weight.shape[0]
#     print("embedding_size: " + str(embedding_size))
#     print("len(tokenizer): " + str(len(tokenizer)))
#     if len(tokenizer) > embedding_size:
#         model.resize_token_embeddings(len(tokenizer))
#
#     print("----------------------------finish loading base model-----------------------------")
#
#     if args.initialize_lora:
#         # 这一项默认是false
#         assert not isinstance(model, PeftModel)
#         lora_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             inference_mode=False,
#             r=args.lora_r,
#             lora_alpha=args.lora_alpha,
#             lora_dropout=args.lora_dropout,
#             target_modules=args.lora_target_modules,
#         )
#         model = get_peft_model(model, lora_config)
#
#     if isinstance(model, PeftModel):
#         # 这个是true
#         print("model.print_trainable_parameters() -> ")
#         model.print_trainable_parameters()
#
#     adam_optimizer_state = None
#     if args.info_type == "grads" and args.gradient_type == "adam":
#         # 要走这个地方获取到warmup后的模型训练情况
#         optimizer_path = os.path.join(args.model_path, "optimizer.bin")
#         adam_optimizer_state = torch.load(optimizer_path, map_location="cpu")["state"]
#
#     # note: 从此处开始正式不一样，开始使用自己的逻辑了
#
#     if args.info_type == "reps":
#         collect_reps(dataloader, model, args.output_path,
#                      max_samples=args.max_samples)
#     elif args.info_type == "grads":
#         # & 走的这里
#         collect_grads(dataloader,
#                       model,
#                       args.output_path,
#                       proj_dim=args.gradient_projection_dimension,
#                       gradient_type=args.gradient_type,
#                       adam_optimizer_state=adam_optimizer_state,
#                       max_samples=args.max_samples)
#     elif args.info_type == "loss":
#         get_loss(dataloader, model, args.output_path)
#
# if __name__ == "__main__":
