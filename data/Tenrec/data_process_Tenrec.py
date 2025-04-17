import csv

import numpy as np


def split_indices(data_length, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    # Set the seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Generate a list of indices
    indices = np.arange(data_length)

    # Shuffle the indices
    np.random.shuffle(indices)

    # Calculate the split indices
    train_idx = int(train_ratio * data_length)
    val_idx = train_idx + int(val_ratio * data_length)

    # Split the indices into training, validation, and testing
    train_indices = indices[:train_idx]
    val_indices = indices[train_idx:val_idx]
    test_indices = indices[val_idx:]

    return train_indices, val_indices, test_indices


# Please download 'QB-video.csv' of the Tenrec dataset.
filename = 'QB-video.csv'  # user_id,item_id,click,follow,like,share,video_category,watching_times,gender,age
user_map = dict()
item_map = dict()
inter_seq = dict()  # interaction sequence
expo_seq = dict()  # exposure data sequence
expo_seq_ = dict()  # exposure data sequence
user_count = dict()
item_count = dict()  # only cal the count of click event
user_tot = item_tot = 1  # id of user and item both start at 1
with open(filename) as csvfile:  # get user_count and item_count
    csv_reader = csv.reader(csvfile)  # read file using csv
    csv_header = next(csv_reader)
    for row in csv_reader:
        user_id = int(row[0])
        item_id = int(row[1])
        click = int(row[2])
        if user_id not in user_map:
            user_map[user_id] = user_tot
            user_count[user_id] = 0
            user_tot += 1
            # inter_seq[user_id] = []
            # expo_seq[user_id] = []
        if item_id not in item_map:
            item_map[item_id] = item_tot
            item_count[item_id] = 0
            item_tot += 1
        if click == 1:
            item_count[item_id] += 1
            user_count[user_id] += 1

with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # read file using csv
    csv_header = next(csv_reader)
    for row in csv_reader:
        user_id = int(row[0])
        item_id = int(row[1])
        click = int(row[2])
        # filter out these users and items in inter_seq and exposure data
        if user_count[user_id] < 2 or item_count[item_id] < 5:
            continue
        user_id = user_map[user_id]
        item_id = item_map[item_id]
        if user_id not in inter_seq:
            inter_seq[user_id] = []
            expo_seq[user_id] = []
            expo_seq_[user_id] = []
        if click == 1:
            inter_seq[user_id].append(item_id)
            expo_seq_[user_id].append((item_id, 1))
        else:
            expo_seq_[user_id].append((item_id, 0))
        expo_seq[user_id].append(item_id)  # click + unclick

tmp_inter_seq = dict()
for user in inter_seq:
    nfeedback = len(inter_seq[user])
    if nfeedback > 50:
        inter_seq[user] = inter_seq[user][:50]
    if nfeedback >= 2:
        tmp_inter_seq[user] = inter_seq[user]
    else:
        expo_seq.pop(user)
        expo_seq_.pop(user)
inter_seq = tmp_inter_seq.copy()
for user in expo_seq:
    nfeedback = len(expo_seq[user])
    if nfeedback > 200:
        expo_seq[user] = expo_seq[user][:200]
        expo_seq_[user] = expo_seq_[user][:200]

# remapping item id
item_remap = dict()
new_item_id = 1
for user in inter_seq:
    new_list = []
    for item in inter_seq[user]:
        if item not in item_remap:
            item_remap[item] = new_item_id
            new_item_id += 1
        item = item_remap[item]
        new_list.append(item)
    inter_seq[user] = new_list
for user in expo_seq:  # fitter out the items only appeared in exposure data
    new_list = []
    # new_list_ = []
    for item in expo_seq[user]:
        if item not in item_remap:
            # expo_seq[user].remove(item)
            continue
        item = item_remap[item]
        new_list.append(item)
        # if item in inter_seq[user]:
        #     new_list_.append((item, 1))
        # else:
        #     new_list_.append((item, 0))
    expo_seq[user] = new_list
    # expo_seq_[user] = new_list_

for user in expo_seq_:
    new_list = []
    for item in expo_seq_[user]:
        if item[0] not in item_remap:  # fitter out the items only appeared in exposure data
            continue
        new_item = item_remap[item[0]]
        new_list.append((new_item, item[1]))
    expo_seq_[user] = new_list

num_instances = sum([len(ilist) for _, ilist in inter_seq.items()])
num_instances_expo = sum([len(ilist) for _, ilist in expo_seq.items()])
print('total user: ', len(inter_seq))
print('total instances: ', num_instances)
print('total items: ', len(item_remap))

print("--- click sequence ---")
maxlen = 0
minlen = 1000000
avglen = 0
for _, ilist in inter_seq.items():
    listlen = len(ilist)
    maxlen = max(maxlen, listlen)
    minlen = min(minlen, listlen)
    avglen += listlen
avglen /= len(inter_seq)
print('max length: ', maxlen)
print('min length: ', minlen)
print('avg length: ', avglen)
print('density: ', num_instances / (len(inter_seq) * len(item_remap)))

print("--- exposure data ---")
maxlen = 0
minlen = 1000000
avglen = 0
for _, ilist in expo_seq.items():
    listlen = len(ilist)
    maxlen = max(maxlen, listlen)
    minlen = min(minlen, listlen)
    avglen += listlen
avglen /= len(inter_seq)
print('max length: ', maxlen)
print('min length: ', minlen)
print('avg length: ', avglen)
print('density: ', num_instances_expo / (len(inter_seq) * len(item_remap)))

# split dataset and write file
inter_seq = [inter_seq[items] for items in inter_seq]  # dict2list
expo_seq = [expo_seq[items] for items in expo_seq]
expo_seq_ = [expo_seq_[items] for items in expo_seq_]
data_len = len(inter_seq)
train_idx = int(data_len * 0.8)
val_idx = train_idx + int(data_len * 0.1)
test_idx = val_idx + int(data_len * 0.1)

train_indices = [i for i in range(1, int(train_idx))]
valid_indices = [i for i in range(train_idx, val_idx)]
test_indices = [i for i in range(val_idx, test_idx)]

train = [inter_seq[i] for i in train_indices]

train_items = set()
for sublist in train:
    train_items.update(sublist)

val = [inter_seq[i] for i in valid_indices]
test = [inter_seq[i] for i in test_indices]


# train data augmentation
def augment(seq_list):
    u2seq = []
    for seq in seq_list:
        for i in range(len(seq), 1, -1):
            u2seq.append(seq[:i])
    return u2seq


def truncate_list(lst):
    # 初始化最后一个1的索引为None
    last_one_index = None

    # 逆向遍历列表找到最后一个1的索引
    # print(lst)
    for index in range(len(lst) - 1, -1, -1):
        if lst[index][1] == 1:
            last_one_index = index
            break
    # 如果找到了1，截取列表到最后一个1的索引
    if last_one_index is not None:
        truncated_lst = lst[:last_one_index + 1]
    else:
        truncated_lst = None  # 如果列表中没有1，返回空列表

    return truncated_lst


def tuple2file(data, dfile, filter_items=None, if_truncated=True):
    with open(dfile, 'w') as f:
        append_items = set()
        for items in data:
            if if_truncated:
                items = truncate_list(items)
            if items is None:
                continue
            if tuple(items) not in append_items:
                # print(items)
                # print(append_items, '\n\n')
                append_items.add(tuple(items))

                if not filter_items is None:
                    output_items = []
                    for item in items:
                        if item[0] in filter_items:
                            output_items.append([item[0], item[1]])
                    if len(output_items) < 2:
                        continue
                else:
                    output_items = items

                for item in output_items:
                    f.write(str(item[0]) + ':' + str(item[1]) + " ")
                f.write('\n')


def tuple2file4RT(data, dfile, filter_items=None):
    with open(dfile, 'w') as f:
        for items in data:

            if not filter_items is None:
                output_items = []
                for item in items:
                    if item[0] in filter_items:
                        output_items.append([item[0], item[1]])
                if len(output_items) < 2:
                    continue
            else:
                output_items = items

            for item in output_items:
                f.write(str(item[0]) + ':' + str(item[1]) + " ")
            f.write('\n')


# train = train

def writetofile(data, dfile, filter_items=None):
    with open(dfile, 'w') as f:
        for items in data:

            if not filter_items is None:
                output_items = []
                for item in items:
                    if item in filter_items:
                        output_items.append(item)
                if len(output_items) < 2:
                    continue
            else:
                output_items = items

            for item in output_items:
                f.write(str(item) + " ")
            f.write('\n')


path = './'
writetofile(train, path + "ori_Train.txt")
writetofile(train, path + "ori_Train.txt")
train = augment(train)
writetofile(train, path + "Train.txt")
writetofile(val, path + "Valid.txt")
writetofile(test, path + "Test.txt")
aug_test = augment(test)
writetofile(aug_test, path + "session_Test.txt")

# 70% exposure data to train the exposure model and remain 30% to evaluation the performance of debias

# 70% data for training exposure model
expo_idx = int(data_len * 0.7)
expo_indices = [i for i in range(1, int(expo_idx))]
expo = [expo_seq[i] for i in expo_indices]
expo_ = [expo_seq_[i] for i in expo_indices]

train_idx = int(expo_idx * 0.8)
val_idx = train_idx + int(expo_idx * 0.1)
test_idx = val_idx + int(expo_idx * 0.1)
train_indices = [i for i in range(1, int(train_idx))]
valid_indices = [i for i in range(train_idx, val_idx)]
test_indices = [i for i in range(val_idx, test_idx)]
expo_train = [expo[i] for i in train_indices]
expo_train_ = [expo_[i] for i in train_indices]
writetofile(expo_train, path + "ori_Exposure_Train.txt")
tuple2file(expo_train_, path + "ori_Exposure_Train4RT.txt")
expo_train = augment(expo_train)
expo_train_ = augment(expo_train_)
expo_valid = [expo[i] for i in valid_indices]
expo_test = [expo[i] for i in test_indices]
expo_valid_ = [expo_[i] for i in valid_indices]
expo_test_ = [expo_[i] for i in test_indices] + expo_valid_
expo_test_ = augment(expo_test_)

writetofile(expo_train, path + "Exposure_Train.txt")
writetofile(expo_valid, path + "Exposure_Valid.txt")
writetofile(expo_test, path + "Exposure_Test.txt")

tuple2file(expo_train_, path + "Exposure_Train4DT.txt")
tuple2file(expo_train_, path + "Exposure_Train4DT_No_Truncated.txt", if_truncated=False)

tuple2file4RT(expo_train_, path + "Exposure_Train4RT.txt")
tuple2file4RT(expo_test_, path + "Exposure_Test4RT.txt")
eval_idx = data_len
eval_indices = [i for i in range(expo_idx, data_len)]
eva = [expo_seq[i] for i in eval_indices]

eval_len = data_len * 0.3
train_idx = int(eval_len * 0.8)
val_idx = train_idx + int(eval_len * 0.1)
test_idx = val_idx + int(eval_len * 0.1)
train_indices = [i for i in range(1, int(train_idx))]
valid_indices = [i for i in range(train_idx, val_idx)]
test_indices = [i for i in range(val_idx, test_idx)]
eval_train = [eva[i] for i in train_indices]
writetofile(eval_train, path + "ori_Evaluation_Train.txt")
eval_train = augment(eval_train)
eval_valid = [eva[i] for i in valid_indices]
eval_test = [eva[i] for i in test_indices]
writetofile(eval_train, path + "Evaluation_Train.txt")
writetofile(eval_valid, path + "Evaluation_Valid.txt")
writetofile(eval_test, path + "Evaluation_Test.txt")

"""
total user:  31722
total instances:  912812
total items:  24653
--- click sequence ---
max length:  50
min length:  2
avg length:  28.775360948237815
density:  0.0011672153875081255
--- exposure data ---
max length:  200
min length:  2
avg length:  54.792131643654244
density:  0.002222534038196335
"""
