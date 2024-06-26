# note 结合了之前的 matching和 selecting data的任务，直接全部完成
# note 支持指定 sample_size，不会一次处理一整个数据集
# note 这段代码是针对 layer的不同设计的，可以指定layer进行处理

import json
import os
import torch

# argparser.add_argument('--output_path', type=str, default="../selected_data", help='The path to the output')

share_config = {
    "DIM": 8192,
    "SEED": 3,
    "train_file_names": ["code_high", "code_medium", "code_low"],
    "target_task_names": ["AQuA", "ASDiv", "ASDiv_Grade_1", "ASDiv_Grade_2", "ASDiv_Grade_3", "ASDiv_Grade_4",
                          "ASDiv_Grade_5", "ASDiv_Grade_6", "GSM", "LeeTCode_submission", "MultiArith", "SVAMP",
                          "olympic_OE_TO_maths_en_COMP", "olympic_OE_TO_physics_en_COMP",
                          "olympic_TP_TO_maths_en_COMP", "olympic_TP_TO_physics_en_COMP"],
    # only select data
    "percentage": 0.05,
    "max_samples": None,

    # sample size
    "sample_size": [5, 10, 50]
}

matching_configs = [
    {
        "ckpts": [4, 8, 12, 16],
        "MODEL_NAME": "llama2-7b",
        "checkpoint_weights": [1.8000000000000004e-05,
                               1.2666666666666665e-05,
                               7.333333333333333e-06,
                               2.0000000000000003e-06],

        # 之前算出来的梯度的位置，训练集+测试集，需要读，记得改model_name
        "base_path": "E:/backup_for_servicer/1_project/project_870_code/",
        "gradient_path": "grads/llama2-7b-p0.05-lora-seed3/{}-ckpt{}-adam/dim8192/all_orig.pt",
        "validation_gradient_path": "grads/llama2-7b-p0.05-lora-seed3/{}-ckpt{}-sgd/dim8192/all_orig.pt",
        # 输出的地址
        "ori_output_path": "E:/backup_for_servicer/1_project/analyse_result/870_code",
        # todo 加上对分层的描述，这个先不急，之后可以有单独设置的参数，粗出现每个层自己做就好了

        # layer config

    },
    {
        # llama2-7b， 1700数据
        "ckpts": [7, 15, 23, 28],
        "MODEL_NAME": "llama2-7b",
        "checkpoint_weights": [1.7777777777777777e-05,
                               1.2222222222222222e-05,
                               6.296296296296296e-06,
                               1.4814814814814815e-06],

        # 之前算出来的梯度的位置，训练集+测试集，需要读，记得改model_name
        "base_path": "E:/backup_for_servicer/1_project/project_1700_code/",
        "gradient_path": "grads/llama2-7b-p0.05-lora-seed3/{}-ckpt{}-adam/layer_from_{}_to_{}/all_orig.pt",
        "validation_gradient_path": "grads/llama2-7b-p0.05-lora-seed3/{}-ckpt{}-sgd/layer_from_{}_to_{}/all_orig.pt",
        # 输出的地址
        "ori_output_path": "E:/backup_for_servicer/1_project/analyse_result/1700_code",
    }
]

N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9, "AQuA": 1, "GSM": 1, "ASDiv": 1, "ASDiv_Grade_1": 1,
              "ASDiv_Grade_2": 1, "ASDiv_Grade_3": 1, "ASDiv_Grade_4": 1, "ASDiv_Grade_5": 1, "ASDiv_Grade_6": 1,
              "GSM": 1, "LeeTCode_submission": 1, "MultiArith": 1, "SVAMP": 1, "olympic_OE_TO_maths_en_COMP": 1,
              "olympic_OE_TO_physics_en_COMP": 1, "olympic_TP_TO_maths_en_COMP": 1, "olympic_TP_TO_physics_en_COMP": 1}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.
    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores


def new_matching(config):
    # 跟之前的逻辑一样，将所有的分数计算出来存起来
    # renormalize the checkpoint weights
    if sum(config["checkpoint_weights"]) != 1:
        s = sum(config["checkpoint_weights"])
        config["checkpoint_weights"] = [i / s for i in config["checkpoint_weights"]]  # 可能会出问题，因为梯度其实差的挺多的

    # todo 后续这部分应该可以由上层指定
    from_layer_list = [i for i in range(0, 32)]  # 0-31
    end_layer_list = [i for i in range(0, 32)]  # 0-31 todo 这一部分之后要可以由上层指定

    # calculate the influence score for each validation task
    # 测试集的名，实际是上列表
    for target_task_name in config["target_task_names"]:
        for layer_idx in range(0, len(from_layer_list)):
            current_layer_str = f"layer_from_{from_layer_list[layer_idx]}_to_{end_layer_list[layer_idx]}"
            for sample_size in config["sample_size"]:
                print("--------------------------------------------")
                mmodel_name = config["MODEL_NAME"]
                print(f"测试集为{target_task_name}, sample_size为 {sample_size}, model为 {mmodel_name}")

                dataset_count = {"top5": {}, "top1": {}, }
                data_id_count = {"top5": {}, "top1": {}, }

                # note 现在才是开始正式的进行一个单位的处理流程，即明确的测试集，输出位置，sample_size
                validation_data_num = torch.load(
                    config["validation_gradient_path"].format(target_task_name,
                                                              config["ckpts"][0],
                                                              from_layer_list[layer_idx],
                                                              end_layer_list[layer_idx])).shape[0]

                # note 对每一个offset进行全流程的处理，每个offset是一个sample_size的数据，每个offset内的数据对应于之前那人写的一个数据集
                for offset in range(0, validation_data_num, sample_size):
                    cur_offset_score = {}
                    for train_file_name in config["train_file_names"]:  # 训练集的名
                        influence_score = 0

                        # 每个checkpoint的分数
                        for i, ckpt in enumerate(config["ckpts"]):
                            # 加载测试集计算出来的梯度，读取数据然后仅提取需要的部分
                            validation_path = config["validation_gradient_path"].format(target_task_name,
                                                                                        ckpt,
                                                                                        from_layer_list[layer_idx],
                                                                                        end_layer_list[layer_idx])
                            validation_info = torch.load(validation_path)
                            end_index = min(offset + sample_size, validation_info.shape[0])
                            validation_info = validation_info[offset:end_index]

                            if not torch.is_tensor(validation_info):
                                validation_info = torch.tensor(validation_info)

                            # shape为 [254, 8192]，第一维的大小对应了eval集合的数据条数
                            validation_info = validation_info.to(device).float()

                            # 加载训练集计算出来的梯度
                            gradient_path = config["gradient_path"].format(train_file_name,
                                                                           ckpt,
                                                                           from_layer_list[layer_idx],
                                                                           end_layer_list[layer_idx])
                            training_info = torch.load(gradient_path)
                            if not torch.is_tensor(training_info):
                                training_info = torch.tensor(training_info)
                            training_info = training_info.to(device).float()  # shape为 [874, 8192]，第一维的大小对应了train集合的数据条数

                            # 计算出当前checkpoint 的分数，即相关性*学习率，大小为 [n_train, n_eval]
                            influence_score += config["checkpoint_weights"][i] * \
                                               calculate_influence_score(training_info=training_info,
                                                                         validation_info=validation_info)

                        # 记录当前这个训练集的分数
                        # max(-1) 在经过均值处理后的张量的最后一个维度（即每个样本对应的各个子任务的平均影响力）上求最大值。这个操作返回两个结果：最大值本身和这些最大值所在的索引。由于代码中使用 [0] 索引，这意味着我们只获取最大值，而不关心这些最大值的具体位置。
                        # 这一步的输出是每个样本在其所有子任务的平均影响力中的最大值。
                        # 最终influence_score 的大小就是 [train_set_size]
                        influence_score = influence_score.reshape(
                            influence_score.shape[0], N_SUBTASKS[target_task_name], -1).mean(-1).max(-1)[0]
                        cur_offset_score[train_file_name] = influence_score

                        # 保存结果，已弃用
                        # output_file = os.path.join(
                        #     config["output_path"], config["MODEL_NAME"], target_task_name,
                        #     f"{train_file_name}_influence_score.pt")
                        # os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        # torch.save(influence_score, str(output_file))
                        # print("Saved influence score to {}".format(output_file))

                    # 对于某一次offset，现在3个数据集都跑完了，可以加载然后进行计算统计了
                    new_select(config, cur_offset_score, dataset_count, data_id_count)
                    print(f"\rfinish offset: {offset}/{validation_data_num}", end="")

                # 一个sample_size 走完了，即offset全部过了一遍，开始存储信息了
                print("\nfinish sample_size: ", sample_size)

                # 确定存储地址
                save_dir = os.path.join(config["ori_output_path"], f"/{sample_size}", config["MODEL_NAME"])
                os.makedirs(save_dir, exist_ok=True)
                count_file_name = f"{target_task_name}_dataset_count_{sample_size}_{current_layer_str}.json"
                id_file_name = f"{target_task_name}_data_id_count_{sample_size}_{current_layer_str}.json"

                with open(os.path.join(save_dir, count_file_name), 'w') as file:
                    json.dump(dataset_count, file, indent=4)
                with open(os.path.join(save_dir, id_file_name), 'w') as file:
                    json.dump(data_id_count, file, indent=4)


def new_select(config, score_dict, dataset_count, data_id_count):
    assert config["percentage"] is not None or config["max_samples"] is not None

    # target_task是一个具体的测试集的名字
    # 处理是每个测试集单独进行的

    # 将之前计算出的所有分数拼接起来，这里是按顺序的，比如前面的一部分是code_high的，中间就是medium的
    all_scores = []
    num_samples = []
    # for score_path in score_paths:
    for train_name in config["train_file_names"]:
        # score = torch.load(score_path, map_location=device)
        score = score_dict[train_name]
        num_samples.append(len(score))  # 分别是每个训练数据集的数据个数
        all_scores.append(score)
    all_scores = torch.cat(all_scores, dim=0)
    total_samples = sum(num_samples)

    # 其实现在已经不用了
    if config["percentage"] is not None:
        config["max_samples"] = int(config["percentage"] * total_samples)

    # sort the scores and output the corresponding data index
    # 生成index，为(0, 1, ... ,873, 0, 1, ... 6237, 0, 1, ...1670) 分别是每个数据集从0到最大的部分
    file_specific_index = torch.cat(
        [torch.arange(line_num) for line_num in num_samples]).to(device)

    # 生成每个条目所属的train dataset的index，为(0, 0, 0... 1,1,1...., 2,2,2...)
    data_from = torch.cat(
        [torch.ones(line_num, dtype=torch.long) * i for i, line_num in enumerate(num_samples)]).to(device)

    # all_scores按分数降序。torch.sort 返回两个张量：一个是排序后的分数 (sorted_scores)，另一个是排序后分数对应的原始索引 (sorted_index)。
    sorted_scores, sorted_index = torch.sort(all_scores, dim=0, descending=True)

    data_from = data_from[sorted_index]  # 数据集
    sorted_index = file_specific_index[sorted_index]  # 这时候sorted_index[i]就变成了排序后的第[i]项在自己所属的数据集内的index

    ccount = 0
    for score, index, data_from_info in zip(sorted_scores, sorted_index, data_from):
        cur_dataset_name = config["train_file_names"][data_from_info.item()]
        cur_data_id = cur_dataset_name + "_" + str(index.item())  # 刚好碰巧了行号+数据集name就是id

        # 保存数据
        if ccount <= total_samples * 0.05:
            dataset_count["top5"][cur_dataset_name] = dataset_count["top5"].get(cur_dataset_name, 0) + 1
            data_id_count["top5"][cur_data_id] = data_id_count["top5"].get(cur_data_id, 0) + 1
        if ccount <= total_samples * 0.01:
            dataset_count["top1"][cur_dataset_name] = dataset_count["top1"].get(cur_dataset_name, 0) + 1
            data_id_count["top1"][cur_data_id] = data_id_count["top1"].get(cur_data_id, 0) + 1
        ccount += 1
        # xxx = config["train_file_names"]
        # file.write(f"{xxx[data_from_info.item()]}, {index.item()}, {round(score.item(), 6)}\n")


if __name__ == "__main__":
    for config in matching_configs:
        # 计算出matching所需要的所有的参数
        config["gradient_path"] = config["base_path"] + config["gradient_path"]
        config["validation_gradient_path"] = config["base_path"] + config["validation_gradient_path"]
        config.update(share_config)
        new_matching(config)
