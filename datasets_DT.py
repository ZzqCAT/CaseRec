"""
@Time       : 2024/6/10 16:42
@File       : datasets_DT.py
@Description: 
"""
import random

import torch
from torch.utils.data import Dataset


class DTDataset(Dataset):

    def __init__(self, data, context_length, encoder_length, relabel=False):
        self.data = data
        self.context_length = context_length
        self.encoder_length = encoder_length
        self.relabel = relabel

    def __getitem__(self, index):
        user_id = index
        this_data = self.data[index]
        items = [item[0] for item in this_data]
        behaviour_data = [item[1] for item in this_data]
        if self.relabel:
            total_targets = []
            for idx in range(len(items)):
                found = False
                for j in range(idx, len(items)):  # 从当前位置开始查找
                    if behaviour_data[j] == 1:
                        total_targets.append(items[j])
                        found = True
                        break
                if not found:
                    total_targets.append(0)  # 如果当前位置及之后没有属性为1的物品，填充None或其他标记

            # print(total_targets)
        else:
            total_targets = items
        # items=[x0, x1, x2, x3, ..., x30]
        # e.g. encoder_length=10, context_length=10
        # states: (batch, context_len, gru_len)
        # e.g. [[x11,x12,...,x20],...,[x20,...,x28,x29]]
        # if items=[x0, x1, ..., x16] => [[x0,..,x5,0,0..,0],[x0,x1, x2,...,x6,0,0,0],...,[x6,x7,...,x15]]
        # seq_len: (batch, context_len, 1)
        # e.g. [[10],[10],[10],...,[10]]
        # actions: (batch, context_len, 1)
        # e.g. [[x21],[x22],...,[x29]]
        # targets: (batch, context_len, 1)
        # e.g. [[x21],[x22],...,[x29]]
        # rtgs: (batch, context_len, 1)
        # e.g. [[10],[9],[8],...,[1]]
        # timesteps: (batch, 1, 1)
        # e.g. [[30]]
        # if items = [x0, x1, x2, x3, x4, x5]
        # states => [[x0,0,..,0],[x0,x1,0,..,0],[x0,x1,x2,0,..,0],...,[x0,x1,x2,x3,x4,0,...,0],[0,0,..0],..,[0,..,0]]
        # seq_len => [[1],[2],[3],[4],[5],[0],[0],...,[0]]
        # actions => [[x1],[x2],[x3],[x4],[x5],[0],[0],...,[0]]
        # targets => [[x1],[x2],[x3],[x4],[x5],[0],[0],...,[0]]
        # rtgs => [[5],[4],[3],[2],[1],[0],[0],...,[0]]
        # timesteps => [[5]]

        input_ids = items[:-1]
        label = items[-1]
        end_id = len(input_ids)
        start_id = max(0, end_id - self.context_length)
        states = []
        seq_len = []
        actions = []
        targets = []
        rewards = []
        behaviours = []
        for index in range(start_id, end_id):
            s_id = max(0, index - self.encoder_length + 1)
            this_items = input_ids[s_id:index + 1]
            seq_len.append(len(this_items))
            actions.append(items[index + 1])
            targets.append(total_targets[index + 1])
            rewards.append(behaviour_data[index + 1])
            states.append(this_items + [0] * (self.encoder_length - len(this_items)))
            behaviours.append(behaviour_data[s_id:index + 1] + [0] * (self.encoder_length - len(this_items)))
        states = states + [[0 for _ in range(self.encoder_length)]] * (self.context_length - len(states))
        behaviours = behaviours + [[0 for _ in range(self.encoder_length)]] * (self.context_length - len(behaviours))
        actions = actions + [0] * (self.context_length - len(actions))
        targets = targets + [0] * (self.context_length - len(targets))
        seq_len = seq_len + [0] * (self.context_length - len(seq_len))
        rewards_as_label = rewards + [-1] * (self.context_length - len(rewards))
        rewards = rewards + [0] * (self.context_length - len(rewards))
        rtgs = [sum(rewards[i:]) for i in range(len(rewards))]
        timesteps = [len(input_ids)]

        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(label, dtype=torch.long),  # label
            torch.tensor(states, dtype=torch.long),  # states
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),  # actions
            torch.tensor(targets, dtype=torch.long).unsqueeze(1),  # targets
            torch.tensor(rtgs, dtype=torch.float32).unsqueeze(1),  # return-to-go
            torch.tensor(timesteps, dtype=torch.int64).unsqueeze(1),  # global timestep
            torch.tensor(seq_len, dtype=torch.long),  # length of each input states
            torch.tensor(behaviours, dtype=torch.long),  # behaviours for items
            torch.tensor(rewards_as_label, dtype=torch.long),  # rewards
            torch.tensor(behaviour_data[-1], dtype=torch.long),  # rewards_label
        )
        return cur_tensors

    def __len__(self):
        return len(self.data)


class DTDataset4MB(Dataset):
    def __init__(self, item_list, behaviour_list, context_length, encoder_length, is_relabel=False,
                 remap=None, target_behavior=4):
        self.item_list = item_list
        self.behaviour_list = behaviour_list
        """
        BEHAVIOR_MAP = {"retail":{'buy': 1, 'pv': 2, 'fav': 3, 'cart': 4},\
        "ijcai":{'buy': 1, 'pv': 2, 'fav': 3, 'cart': 4},\
        "yelp":{'pos': 1, 'neg': 2, 'neutral': 3, 'tip': 4}}
        """
        self.context_length = context_length
        self.encoder_length = encoder_length
        self.is_relabel = is_relabel
        self.target_behavior = target_behavior
        if remap is not None:
            new_behaviour_list = []
            for behaviours in behaviour_list:
                sub_list = []
                for b in behaviours:
                    sub_list.append(remap[b])
                new_behaviour_list.append(sub_list)
            self.behaviour_list = new_behaviour_list

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, index):
        user_id = index
        items = self.item_list[index]
        behaviour_data = self.behaviour_list[index]
        if self.is_relabel:
            total_targets = []
            for idx in range(len(items)):
                found = False
                for j in range(idx, len(items)):  # 从当前位置开始查找
                    if behaviour_data[j] == self.target_behavior:
                        total_targets.append(items[j])
                        found = True
                        break
                if not found:
                    total_targets.append(0)  # 如果当前位置及之后没有属性为1的物品，填充None或其他标记

            # print(total_targets)
        else:
            total_targets = items

        input_ids = items[:-1]
        label = items[-1]
        end_id = len(input_ids)
        start_id = max(0, end_id - self.context_length)
        states = []
        seq_len = []
        actions = []
        targets = []
        rewards = []
        behaviours = []
        for index in range(start_id, end_id):
            s_id = max(0, index - self.encoder_length + 1)
            this_items = input_ids[s_id:index + 1]
            seq_len.append(len(this_items))
            actions.append(items[index + 1])
            targets.append(total_targets[index + 1])
            rewards.append(behaviour_data[index + 1])
            states.append(this_items + [0] * (self.encoder_length - len(this_items)))
            behaviours.append(behaviour_data[s_id:index + 1] + [0] * (self.encoder_length - len(this_items)))
        states = states + [[0 for _ in range(self.encoder_length)]] * (self.context_length - len(states))
        behaviours = behaviours + [[0 for _ in range(self.encoder_length)]] * (self.context_length - len(behaviours))
        actions = actions + [0] * (self.context_length - len(actions))
        targets = targets + [0] * (self.context_length - len(targets))
        seq_len = seq_len + [0] * (self.context_length - len(seq_len))
        rewards_as_label = rewards + [-1] * (self.context_length - len(rewards))
        rewards = rewards + [0] * (self.context_length - len(rewards))
        rtgs = [sum(rewards[i:]) for i in range(len(rewards))]
        timesteps = [len(input_ids)]

        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(label, dtype=torch.long),  # label
            torch.tensor(states, dtype=torch.long),  # states
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),  # actions
            torch.tensor(targets, dtype=torch.long).unsqueeze(1),  # targets
            torch.tensor(rtgs, dtype=torch.float32).unsqueeze(1),  # return-to-go
            torch.tensor(timesteps, dtype=torch.int64).unsqueeze(1),  # global timestep
            torch.tensor(seq_len, dtype=torch.long),  # length of each input states
            torch.tensor(behaviours, dtype=torch.long),  # behaviours for items
            torch.tensor(rewards_as_label, dtype=torch.long),  # rewards
            torch.tensor(behaviour_data[-1], dtype=torch.long),  # rewards_label
        )
        return cur_tensors





class AugDataset(Dataset):
    def __init__(self, data, ratio, aug_length, encoder_length, context_length, item_size, replace_strategy='random'):
        self.ratio = ratio
        self.item_size = item_size
        self.aug_length = aug_length
        self.replace_strategy = replace_strategy
        self.encoder_length = encoder_length
        self.context_length = context_length
        self.ori_data = self.clean_data(data)
        self.data = self.random_sample()

    def clean_data(self, data):
        return [seq for seq in data if len(seq) > self.encoder_length + self.aug_length + self.encoder_length]

    def random_sample(self):
        total_elements = sum(len(sublist) for sublist in self.ori_data)

        # 计算目标抽样元素数量
        target_sample_size = int(total_elements * self.ratio)

        sampled_sublists = []
        current_sample_size = 0

        # 随机排列子列表
        shuffled_sublists = random.choices(self.ori_data, k=int((1 + self.ratio) * len(self.ori_data)))

        # 按序添加子列表直到接近目标数量
        for sublist in shuffled_sublists:
            begin = random.randint(self.encoder_length + self.context_length - 2, len(sublist) - 1)
            random_list = sublist[begin - (self.encoder_length + self.context_length - 2):begin + self.aug_length + 1]
            sampled_sublists.append(random_list + [(0, 0)] * (
                    self.encoder_length + self.context_length + self.aug_length - 1 - len(random_list)))
            current_sample_size += self.aug_length

            # 如果已经达到或超过目标抽样元素数量，则停止
            if current_sample_size >= target_sample_size:
                break

        return sampled_sublists

    def random_replace(self, seq):
        seq[-self.aug_length:] = [random.randint(1, self.item_size - 1) for _ in range(self.aug_length)]
        return seq

    def __getitem__(self, index):
        seq = self.data[index]
        items = [item[0] for item in seq]
        # print(len(items))
        behaviours_data = [item[1] for item in seq][:-self.aug_length]
        ori_seq = seq[:-self.aug_length]
        if self.replace_strategy == 'random':
            items = self.random_replace(items)
        actions = items[-(self.aug_length + self.context_length - 1):]
        begin = -(self.aug_length + self.context_length + self.encoder_length - 1)
        states = []
        for i in range(self.aug_length + self.context_length - 1):
            states.append(items[begin + i:begin + i + self.encoder_length])
        behaviours = []
        for i in range(self.context_length):
            behaviours.append(behaviours_data[i:i + self.encoder_length])
        timesteps = [self.encoder_length + i for i in range(self.aug_length)]
        seq_len = [self.encoder_length for _ in range(self.aug_length + self.context_length - 1)]
        # print(torch.tensor(ori_seq, dtype=torch.long).shape)
        # print(torch.tensor(actions, dtype=torch.long).shape)
        # print(torch.tensor(states, dtype=torch.long).shape)
        cur_tensors = (
            torch.tensor(ori_seq, dtype=torch.long),  # [(item0, behaviour0),...,(item18, behaviour18)]
            # [item10,item11,...,item18,item19*,...,item28*] fixed  [B,cl+al-1,1]
            torch.tensor(actions, dtype=torch.long).unsqueeze(-1),
            # [[item0,item1,...,item9],...,[item18,item19*,...,item27*]] fixed [B,cl+al-1,el]
            torch.tensor(states, dtype=torch.long),
            torch.tensor(timesteps, dtype=torch.long).unsqueeze(-1),  # [10,11,12,...19]
            torch.tensor(seq_len, dtype=torch.long),  # [10,10,10,...,10] fixed [B,cl+al-1]
            torch.tensor(behaviours, dtype=torch.long),  # [[beh0,beh1,...,beh9],...,[beh9,...,beh18]]
        )
        # for t in cur_tensors:
        #     print(t.shape)
        # print('----',torch.tensor(timesteps, dtype=torch.long).unsqueeze(-1).shape) torch.Size([10, 1])
        return cur_tensors

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    context_length = 10
    #
    # i = [1, 2, 3, 4, 5]  # 物品id
    # b = [0, 1, 0, 0, 1]  # 物品对应的属性（0或1）
    #
    # # 生成一个长度一样的list l
    # l = []
    # for idx in range(len(i)):
    #     found = False
    #     for j in range(idx, len(i)):  # 从当前位置开始查找
    #         if b[j] == 1:
    #             l.append(i[j])
    #             found = True
    #             break
    #     if not found:
    #         l.append(0)  # 如果当前位置及之后没有属性为1的物品，填充None或其他标记
    #
    # print(l)

    # encoder_length = 10
    # items = [i for i in range(16)]
    # input_ids = items[:-1]
    # end_id = len(input_ids)
    # start_id = max(0, end_id - context_length)
    # states = []
    # seq_len = []
    # actions = []
    # rewards = []
    # for index in range(start_id, end_id):
    #     s_id = max(0, index - encoder_length + 1)
    #     this_items = input_ids[s_id:index + 1]
    #     seq_len.append(len(this_items))
    #     actions.append(items[index + 1])
    #     rewards.append(1)
    #     states.append(this_items + [0] * (encoder_length - len(this_items)))
    # states = states + [[0 for _ in range(encoder_length)]] * (context_length - len(states))
    # actions = actions + [0] * (context_length - len(actions))
    # seq_len = seq_len + [0] * (context_length - len(seq_len))
    # rewards = rewards + [0] * (context_length - len(rewards))
    # rtgs = [sum(rewards[i:]) for i in range(len(rewards))]
    # timesteps = [len(input_ids)]
    # for s in states:
    #     print(s)
    # print(seq_len)
