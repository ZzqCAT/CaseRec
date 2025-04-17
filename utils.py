# -*- coding: utf-8 -*-
import csv
import pickle
import warnings

import numpy as np
import math
import random
import os
from scipy.sparse import csr_matrix
from collections import Counter

import torch
from torch.cuda import device_count, get_device_capability, get_device_name
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_gpu_capability():
    incorrect_binary_warn = """
    Found GPU%d %s which requires CUDA_VERSION >= %d to
     work properly, but your PyTorch was compiled
     with CUDA_VERSION %d. Please install the correct PyTorch binary
     using instructions from https://pytorch.org
    """

    old_gpu_warn = """
    Found GPU%d %s which is of cuda capability %d.%d.
    PyTorch no longer supports this GPU because it is too old.
    The minimum cuda capability supported by this library is %d.%d.
    """

    # if torch.version.cuda is not None:  # on ROCm we don't want this check
    #     CUDA_VERSION = torch._C._cuda_getCompiledVersion()
    #     for d in range(device_count()):
    #         capability = get_device_capability(d)
    #         major = capability[0]
    #         minor = capability[1]
    #         name = get_device_name(d)
    #         current_arch = major * 10 + minor
    #         min_arch = min((int(arch.split("_")[1]) for arch in torch.cuda.get_arch_list()), default=35)
    #         if current_arch < min_arch:
    #             warnings.warn(old_gpu_warn.format(d, name, major, minor, min_arch // 10, min_arch % 10))
    #             return False
    #         elif CUDA_VERSION <= 9000 and major >= 7 and minor >= 5:
    #             warnings.warn(incorrect_binary_warn % (d, name, 10000, CUDA_VERSION))
    #             return False
    return torch.cuda.is_available()


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')


def neg_sample(target, item_size):
    item = random.randint(1, item_size - 1)
    while item == target:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def generate_rating_matrix(user_seq, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    num_users = len(user_seq)
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def get_user_seqs(data_file):
    item_counter = Counter()
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    user_index = 0
    for line in lines:
        items = line.strip().split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
        item_counter.update(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2
    return user_seq, max_item, item_counter


def get_user_seqs_from_expo(data_file):
    item_counter = Counter()
    lines = open(data_file).readlines()
    user_seq = []
    ori_user_seq = []
    item_set = set()
    user_index = 0
    max_len = 0
    for line in lines:
        items = line.strip().split(' ')
        current_line = []
        counter_items = []
        max_len = max(len(items), max_len)
        for pair in items:
            a, b = pair.split(':')
            if b == '1':
                current_line.append(int(a))
                counter_items.append(int(a))
        items = current_line
        user_seq.append(items)
        ori_user_seq.append(counter_items)
        item_set = item_set | set(counter_items)
        item_counter.update(counter_items)
    max_item = max(item_set)
    # print(f'length of item set = {len(item_set)}')

    num_users = len(lines)
    num_items = max_item + 2
    return ori_user_seq, user_seq, max_item, max_len, item_counter


def get_user_seqs4DT(data_file):
    item_counter = Counter()
    lines = open(data_file).readlines()
    user_seq = []
    ori_user_seq = []
    item_set = set()
    user_index = 0
    max_len = 0
    for line in lines:
        items = line.strip().split(' ')
        current_line = []
        counter_items = []
        max_len = max(len(items), max_len)
        for pair in items:
            if ':' in pair:
                a, b = pair.split(':')
                current_line.append((int(a), int(b)))
                counter_items.append(int(a))
            else:
                a = pair
                current_line.append((int(a), 1))
                counter_items.append(int(a))
        items = current_line
        user_seq.append(items)
        ori_user_seq.append(counter_items)
        item_set = item_set | set(counter_items)
        item_counter.update(counter_items)
    max_item = max(item_set)
    # print(f'length of item set = {len(item_set)}')

    num_users = len(lines)
    num_items = max_item + 2
    return ori_user_seq, user_seq, max_item, max_len, item_counter


def read_data4DT(runs_dir, data_dir, post_fix):
    runs_file = os.path.join(runs_dir, post_fix + '.pkl')
    data_file = os.path.join(data_dir, post_fix + '.txt')
    if not os.path.exists(os.path.dirname(runs_file)):
        os.makedirs(os.path.dirname(runs_file))
    if not os.path.exists(runs_file):
        print(f'load data from {data_file}...')
        data, data4dt, max_item, max_len, item_counter = get_user_seqs4DT(data_file)
        objects_to_save = (data, data4dt, max_item, max_len, item_counter)
        with open(runs_file, 'wb') as f:
            pickle.dump(objects_to_save, f)
        return data, data4dt, max_item, max_len, item_counter
    else:
        print(f'load data from {runs_file}...')
        with open(runs_file, 'rb') as f:
            loaded_objects = pickle.load(f)
        data, data4dt, max_item, max_len, item_counter = loaded_objects
        return data, data4dt, max_item, max_len, item_counter


def apt_at_k(actual, niche_set, predicted, topk, pred=None):
    num_users = len(predicted)
    apt = 0
    apt_p = 0
    for i in range(num_users):
        label = actual[i][0]
        pred_set = set(predicted[i][:topk])
        apt += len(niche_set & pred_set) / float(len(pred_set))
        if pred is not None:
            apt_p += len(niche_set & pred_set) / float(len(pred_set)) / pred[label]

    return apt / num_users, apt_p / num_users


def coverage_at_k(predicted, topk):
    cover_set = set()
    num_users = len(predicted)
    for i in range(num_users):
        pred_set = set(predicted[i][:topk])
        cover_set.update(pred_set)

    return len(cover_set)


def recall_at_k(actual, predicted, topk, pred=None):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        label = actual[i][0]
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            if pred is not None:
                sum_recall += len(act_set & pred_set) / float(len(act_set)) / pred[label]
                true_users += 1 / pred[label]
            else:
                sum_recall += len(act_set & pred_set) / float(len(act_set))
                true_users += 1
    return sum_recall / true_users


def ndcg_k(actual, predicted, topk, pred=None):
    res = 0
    true_users = 0
    for user_id in range(len(actual)):
        label = actual[user_id][0]
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        if pred is not None:
            res += dcg_k / idcg / pred[label]
            true_users += 1 / pred[label]
        else:
            res += dcg_k / idcg
            true_users += 1
    return res / float(true_users)


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def augment_data(seq_list, min_len=1):
    u2seq = []
    for seq in seq_list:
        if min_len >= len(seq):
            u2seq.append(seq)
        else:
            for i in range(len(seq), min_len, -1):
                u2seq.append(seq[:i])
    return u2seq


def count_lines(file_path):
    """Quickly count the number of lines in a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)


def read_tsv_data(file_path):
    user_ids = []
    sequences = []
    behavior_sequences = []
    total_lines = count_lines(file_path) - 1  # Adjust for header if present
    max_id = 0
    max_len = 0

    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')

        # Skip the header or any initial lines if necessary
        next(reader)  # Uncomment or remove based on whether your file has a header

        for row in tqdm(reader, total=total_lines, desc="Processing"):
            user_id = int(row[0])
            item_id_list = list(map(int, row[1].split() + [row[2]]))
            behavior_list = list(map(int, map(float, row[3].split())))
            max_id = max(max_id, max(item_id_list))
            max_len = max(max_len, len(item_id_list))
            user_ids.append(user_id)
            sequences.append(item_id_list)
            behavior_sequences.append(behavior_list)
    return user_ids, sequences, behavior_sequences, max_id, max_len
