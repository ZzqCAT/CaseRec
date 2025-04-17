"""
@Time       : 2024/6/25 16:54
@File       : run_augmentation.py
@Description: augment the dataset by perturbing the action
"""
import datetime
import json
import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets_DT import AugDataset
from model_DT import RewardTransformer, GPTConfig, GPT
from utils import check_path, set_seed, read_data4DT
from trainer_DT import RTAugmenter


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--runs_dir', default='runs/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='ZhihuRec', type=str)

    # model args for RT
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", default="RT", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--model_type', default="reward_conditioned", type=str)  # gelu relu
    parser.add_argument('--context_length', default=10, type=int)
    parser.add_argument('--encoder_length', default=10, type=int)
    parser.add_argument('--max_timestep', default=161, type=int)
    parser.add_argument('--share_encoder', action='store_true')
    parser.add_argument("--no_cuda", action="store_true")

    # model args for recommender model
    parser.add_argument("--rec_hidden_size", type=int, default=64, help="hidden size of transformer model")

    # augment args
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--aug_length", default=10, type=int)
    parser.add_argument("--item_size", type=int, required=True)
    parser.add_argument("--recommendation_item", type=int, default=-1)
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--augment_strategy", type=str, default="random_exposed")
    parser.add_argument("--replace_strategy", type=str, default="random")
    parser.add_argument("--recommender_path", type=str)
    args = parser.parse_args()

    if args.augment_strategy == 'bootstrap':
        args.replace_strategy = 'non-replace'

    if args.recommendation_item == -1:
        args.recommendation_item = args.item_size

    set_seed(args.seed)
    check_path(args.output_dir)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = True
    # save model args
    time_stamp = datetime.datetime.now()

    args.data_file = args.data_dir + args.data_name
    augment_save_path = os.path.join(args.data_file,
                                     f'augment_exposure-ratio{args.ratio}-strategy_{args.augment_strategy}.txt')
    args.runs_dir = args.runs_dir + args.data_name
    train_data, train_data4dt, max_item, max_len, _ = read_data4DT(args.runs_dir, args.data_file,
                                                                   'ori_Exposure_Train4RT')

    output_path = os.path.join(args.output_dir, f'{args.model_name}', f'{args.data_name}',
                               f'augmentation-{time_stamp}.txt')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    args.log_file = output_path

    ori_dataset = AugDataset(data=train_data4dt, context_length=args.context_length, encoder_length=args.encoder_length,
                             aug_length=args.aug_length, item_size=args.item_size, ratio=args.ratio,
                             replace_strategy=args.replace_strategy)
    train_sampler = RandomSampler(ori_dataset)
    dataloader = DataLoader(ori_dataset, sampler=train_sampler, batch_size=args.batch_size)

    dt_config = GPTConfig(args.item_size, args.context_length * 2, n_layer=args.nlayers, n_head=args.nhead,
                          n_embd=args.hidden_size, max_timestep=args.max_timestep, model_type=args.model_type,
                          share_encoder=args.share_encoder, context_length=args.context_length)
    model = RewardTransformer(dt_config).cuda()
    model.load_state_dict(torch.load(args.model_path))
    if args.recommender_path is not None:
        rec_config = GPTConfig(args.recommendation_item, args.context_length * 3, n_layer=args.nlayers, n_head=args.nhead,
                               n_embd=args.rec_hidden_size, max_timestep=args.max_timestep, model_type=args.model_type,
                               share_para4decoder=True, share_encoder=args.share_encoder,
                               context_length=args.context_length)
        recommendation_model = GPT(rec_config).cuda()
        recommendation_model.load_state_dict(torch.load(args.recommender_path))
    else:
        recommendation_model = None
    augmenter = RTAugmenter(model, dataloader, args, augment_strategy=args.augment_strategy,
                            recommendation_model=recommendation_model)

    augmenter.augment()
    augmenter.save_data(augment_save_path)


if __name__ == '__main__':
    main()
