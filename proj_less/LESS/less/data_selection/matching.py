import argparse
import os

import torch

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The name of the target tasks")
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{} ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')

args = argparser.parse_args()

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


# renormalize the checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i / s for i in args.checkpoint_weights]  # 可能会出问题，因为梯度其实差的挺多的

# calculate the influence score for each validation task
for target_task_name in args.target_task_names:  # 测试集的名，实际是上列表
    for train_file_name in args.train_file_names:  # 训练集的名

        print(f"train为 {train_file_name}, eval为 {target_task_name}")
        influence_score = 0

        for i, ckpt in enumerate(args.ckpts):
            # 加载测试集计算出来的梯度
            validation_path = args.validation_gradient_path.format(target_task_name, ckpt)
            validation_info = torch.load(validation_path)
            if not torch.is_tensor(validation_info):
                validation_info = torch.tensor(validation_info)
            validation_info = validation_info.to(device).float()  # shape为 [254, 8192]，第一维的大小对应了eval集合的数据条数

            # 加载训练集计算出来的梯度
            gradient_path = args.gradient_path.format(train_file_name, ckpt)
            training_info = torch.load(gradient_path)
            if not torch.is_tensor(training_info):
                training_info = torch.tensor(training_info)
            training_info = training_info.to(device).float()  # shape为 [874, 8192]，第一维的大小对应了train集合的数据条数

            # 计算出当前checkpoint 的分数，即相关性*学习率，大小为 [n_train, n_eval]
            influence_score += args.checkpoint_weights[i] * \
                               calculate_influence_score(
                                   training_info=training_info, validation_info=validation_info)

        print("influce_Score shape: " + str(influence_score.shape))
        influence_score = influence_score.reshape(
            influence_score.shape[0], N_SUBTASKS[target_task_name], -1).mean(-1).max(-1)[0]

        output_dir = os.path.join(args.output_path, target_task_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(
            args.output_path, target_task_name, f"{train_file_name}_influence_score.pt")
        torch.save(influence_score, output_file)
        print("Saved influence score to {}".format(output_file))
