import argparse
import os

import torch


def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_file_names', type=str,
                           nargs='+', 
                           help='The path to the score file')
    argparser.add_argument('--train_files', type=str, 
                           nargs='+',
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--target_task_names', type=str,
                           nargs='+', help='The name of the target task')
    argparser.add_argument('--output_path', type=str,
                           default="../selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=None,
                           help='The percentage of the data to be selected')

    args = argparser.parse_args()
    return args


def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count


if __name__ == "__main__":
    args = parse_args()
    # import ipdb; ipdb.set_trace()
    assert len(args.train_file_names) == len(args.train_files)
    assert args.percentage is not None or args.max_samples is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train_files = len(args.train_file_names)

    # targer_task是一个列表
    # 处理是每个测试集单独进行的
    for target_task in args.target_task_names:  # "AQuA ASDiv ASDiv_Grade_1 ASDiv_Grade_2"
        output_path = os.path.join(args.output_path, target_task)   # ../selected_data/ {target_task}
        score_paths = [os.path.join(output_path, f"{train_name}_influence_score.pt") for train_name in args.train_file_names]  # 计算出的分数

        num_samples = []
        for score_path in score_paths:
            num_samples.append(len(torch.load(score_path, map_location=device)))    # 分别是每个训练数据集的数据个数
        cumsum_num_samples = torch.cumsum(torch.tensor(num_samples), dim=0)     # 累加和

        # 确定需要sample的数量
        total_samples = sum(num_samples)
        if args.percentage is not None: # percentage=0.05
            args.max_samples = int(args.percentage * total_samples)
            data_amount_name = f"p{args.percentage}"
        else:
            data_amount_name = f"num{args.max_samples}"

        # 将之前计算出的所有分数拼接起来，这里是按顺序的，比如前面的一部分是code_high的，中间就是medium的
        all_scores = []
        for score_path, train_file in zip(score_paths, args.train_files):
            score = torch.load(score_path, map_location=device)
            all_scores.append(score)
        all_scores = torch.cat(all_scores, dim=0)

        # sort the scores and output the corresponding data index
        # 生成index，为(0, 1, ... ,873, 0, 1, ... 6237, 0, 1, ...1670) 分别是每个数据集从0到最大的部分
        file_specific_index = torch.cat(
            [torch.arange(line_num) for line_num in num_samples]).to(device)

        # 生成每个条目所属的train dataset的index，为(0, 0, 0... 1,1,1...., 2,2,2...)
        data_from = torch.cat([torch.ones(line_num, dtype=torch.long) * i for i, line_num in enumerate(num_samples)]).to(device)
        
        # all_scores按分数降序。torch.sort 返回两个张量：一个是排序后的分数 (sorted_scores)，另一个是排序后分数对应的原始索引 (sorted_index)。
        sorted_scores, sorted_index = torch.sort(all_scores, dim=0, descending=True)
        sorted_score_file = os.path.join(output_path, f"sorted.csv")


        data_from = data_from[sorted_index]
        sorted_index = file_specific_index[sorted_index]    # 这时候sorted_index[i]就变成了排序后的第[i]项在自己所属的数据集内的index
        # data_from = data_from[sorted_index]       # 应该是代码bug，这一行应该放上面

        if not os.path.exists(sorted_score_file):
            with open(sorted_score_file, 'w', encoding='utf-8') as file:
                file.write("file name, index, score\n")
                for score, index, name in zip(sorted_scores, sorted_index, data_from):
                    file.write(
                        f"{args.train_file_names[name.item()]}, {index.item()}, {round(score.item(), 6)}\n")


        topk_scores, topk_indices = torch.topk(
            all_scores.float(), args.max_samples, dim=0, largest=True)

        all_lines = []
        for i, train_file in enumerate(args.train_files):
            with open(train_file, 'r', encoding='utf-8', errors='ignore') as file:
                all_lines.append(file.readlines()[:num_samples[i]])

        final_index_list = sorted_index[:args.max_samples].tolist()
        final_data_from = data_from[:args.max_samples].tolist()
        with open(os.path.join(output_path, f"top_{data_amount_name}.jsonl"), 'w', encoding='utf-8', errors='ignore') as file:
            for index, data_from in zip(final_index_list, final_data_from):
                try:
                    file.write(all_lines[data_from][index])
                except:
                    import pdb
                    pdb.set_trace()
