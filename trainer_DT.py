"""
@Time       : 2024/6/9 18:20
@File       : trainer_DT.py
@Description:
"""
import math

import numpy as np
import tqdm
import torch
from torch.optim import Adam
from optim import ScheduledOptim
from utils import recall_at_k, ndcg_k, check_gpu_capability, coverage_at_k, apt_at_k
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


class DTTrainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader, test_dataloader, args, evaluation_model=None, niche_set=None):
        if niche_set is None:
            niche_set = set()
        self.args = args
        self.cuda_condition = check_gpu_capability() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model
        self.evaluation_model = evaluation_model
        if self.cuda_condition:
            self.model.cuda()
            if evaluation_model is not None:
                self.evaluation_model.cuda()
        # Setting the train, valid, and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        raw_model = model.module if hasattr(self.model, "module") else model
        self.optim = raw_model.configure_optimizers(args)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        with open(self.args.log_file, 'a') as f:
            f.write("Total Parameters:" + str(sum([p.nelement() for p in self.model.parameters()])) + '\n')
        self.eval_pred = None
        self.niche_set = niche_set
        # lr
        self.tokens = 0

    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            avg_loss = 0.0
            for i, batch in rec_data_iter:
                self.optim.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                _, label, states, actions, targets, rtgs, timesteps, seq_len, behaviours, _, _ = batch
                logits, loss = self.model(states=states, actions=actions, seq_len=seq_len, behaviours=behaviours,
                                          targets=targets, rtgs=rtgs, timesteps=timesteps)
                loss.backward()
                self.optim.step()

                if self.args.lr_decay:
                    self.tokens += (actions >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < self.args.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, self.args.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - self.args.warmup_tokens) / float(
                            max(1, self.args.final_tokens - self.args.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = self.args.learning_rate * lr_mult
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.args.learning_rate

                avg_loss += loss.item()
            post_fix = {
                "epoch": epoch,
                "loss ": '{:.4f}'.format(avg_loss / len(rec_data_iter)),
                'lr   ': lr
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print('Train:', str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

            '''Debug'''
            return avg_loss / len(rec_data_iter)
        else:
            self.model.eval()
            pred_list = None
            answer_list = None
            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, label, states, actions, targets, rtgs, timesteps, seq_len, behaviours, _, _ = batch
                rating_pred = self.model.predict(states=states, actions=actions, behaviours=behaviours, seq_len=seq_len,
                                                 rtgs=rtgs,
                                                 timesteps=timesteps, temperature=1)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = label.cpu().data
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, label.cpu().data, axis=0)
            answer_list = answer_list.tolist()
            answer_list = [[answer] for answer in answer_list]
            pred_list = pred_list.tolist()
            return self.get_full_sort_score(epoch, answer_list, pred_list)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg, coverage, apt, apt_p = [], [], [], [], []
        # epoch=0 means at testing stage now
        eval_pred = self.eval_pred if self.args.debias_evaluation_k > 0 else None
        for k in [5, 10, 20]:
            recall.append(recall_at_k(answers, pred_list, k, eval_pred))
            ndcg.append(ndcg_k(answers, pred_list, k, eval_pred))
            coverage.append(coverage_at_k(pred_list, k) / self.args.item_size)
            _apt, _apt_p = apt_at_k(answers, self.niche_set, pred_list, k, eval_pred)
            apt.append(_apt)
            apt_p.append(_apt_p)

        post_fix = {
            "Epoch": epoch,
            "Expo_eval: K = ": self.args.debias_evaluation_k,
            "Re@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "Re@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "Re@20": '{:.4f}'.format(recall[2]), "NDCG@20": '{:.4f}'.format(ndcg[2]),
            "coverage@5": '{:.4f}'.format(coverage[0]), "apt@5": '{:.4f}'.format(apt[0]),
            "coverage@10": '{:.4f}'.format(coverage[1]), "apt@10": '{:.4f}'.format(apt[1]),
            "coverage@20": '{:.4f}'.format(coverage[2]), "apt@20": '{:.4f}'.format(apt[2]),
            "apt_p@5": '{:.4f}'.format(apt_p[0]), "apt_p@10": '{:.4f}'.format(apt_p[1]),
            "apt_p@20": '{:.4f}'.format(apt_p[2]),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def load_evaluation(self, file_name):
        self.evaluation_model.load_state_dict(torch.load(file_name))

    def evaluation_pred(self, ori_test_dataloader):
        dataloader = ori_test_dataloader
        str_code = "calculate_evaluation_pred"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s" % (str_code),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        user_tot = 0
        assert self.evaluation_model is not None
        self.evaluation_model.eval()
        for i, batch in rec_data_iter:
            # 0. batch_data will be sent into the device (GPU or cpu)
            batch = tuple(t.to(self.device) for t in batch)
            _, sasrec_input_ids, rec_answer, seq_position, target_neg, gru_input_ids, seq_len = batch
            batch_size = sasrec_input_ids.shape[0]
            user_tot += batch_size
            with torch.no_grad():
                seq_padding = torch.tensor(0, device=self.device, requires_grad=False).repeat(
                    self.args.exposure_max_length - self.args.max_length).unsqueeze(0).repeat(
                    batch_size, 1)
                evaluation_gru_input_ids = torch.cat((gru_input_ids, seq_padding), dim=1)
                evaluation_seq_len = seq_len.detach()
                evaluation_sasrec_input_ids = torch.cat((seq_padding, sasrec_input_ids), dim=1)
                evaluation_seq_position = torch.cat((seq_padding, seq_position), dim=1)
                exp_rating_pred = self.evaluation_model.predict(evaluation_sasrec_input_ids, evaluation_seq_position,
                                                                evaluation_gru_input_ids, evaluation_seq_len)
                exp_rating_pred = exp_rating_pred.sum(dim=0)
                if self.eval_pred is None:
                    self.eval_pred = exp_rating_pred.clone()
                else:
                    self.eval_pred += exp_rating_pred
        assert user_tot != 0
        self.eval_pred = torch.div(self.eval_pred, user_tot)
        self.eval_pred = torch.pow(self.eval_pred, self.args.debias_evaluation_k).tolist()


class RewardTrainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader, test_dataloader, args):
        self.args = args
        self.cuda_condition = check_gpu_capability() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model
        if self.cuda_condition:
            self.model.cuda()
        # Setting the train, valid, and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        raw_model = model.module if hasattr(self.model, "module") else model
        self.optim = raw_model.configure_optimizers(args)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        with open(self.args.log_file, 'a') as f:
            f.write("Total Parameters:" + str(sum([p.nelement() for p in self.model.parameters()])) + '\n')
        self.eval_pred = None
        # lr
        self.tokens = 0

    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            avg_loss = 0.0
            for i, batch in rec_data_iter:
                self.optim.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                _, label, states, actions, targets, rtgs, timesteps, seq_len, behaviours, rewards, _ = batch
                logits, loss = self.model(states=states, actions=actions, seq_len=seq_len, behaviours=behaviours,
                                          targets=rewards, timesteps=timesteps)
                loss.backward()
                self.optim.step()
                if self.args.lr_decay:
                    self.tokens += (actions >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < self.args.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, self.args.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - self.args.warmup_tokens) / float(
                            max(1, self.args.final_tokens - self.args.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = self.args.learning_rate * lr_mult
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.args.learning_rate

                avg_loss += loss.item()
            post_fix = {
                "epoch": epoch,
                "loss ": '{:.4f}'.format(avg_loss / len(rec_data_iter)),
                'lr   ': lr
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print('Train:', str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

            '''Debug'''
            return avg_loss / len(rec_data_iter)
        else:
            self.model.eval()
            pred_list = None
            label_list = None
            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, label, states, actions, _, _, timesteps, seq_len, behaviours, rewards, rewards_label = batch
                predict_rewards = self.model.predict(states=states, actions=actions, seq_len=seq_len,
                                                     behaviours=behaviours, timesteps=timesteps)
                predict_rewards = predict_rewards.cpu().numpy()
                rewards_label = rewards_label.cpu().numpy()
                if i == 0:
                    pred_list = predict_rewards
                    label_list = rewards_label
                else:
                    pred_list = np.append(pred_list, predict_rewards, axis=0)
                    label_list = np.append(label_list, rewards_label, axis=0)
            return self.get_full_evaluate_score(epoch, label_list, pred_list)

    def get_full_evaluate_score(self, epoch, labels, preds):
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        conf_matrix = confusion_matrix(labels, preds)
        post_fix = {
            "Epoch": epoch,
            "Expo_eval: K = ": self.args.debias_evaluation_k,
            "accuracy": '{:.4f}'.format(accuracy), "precision": '{:.4f}'.format(precision),
            "recall": '{:.4f}'.format(recall), "f1": '{:.4f}'.format(f1), "conf_matrix": str(conf_matrix)
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [accuracy, precision, recall, f1], str(post_fix)


class RTAugmenter:
    def __init__(self, model, aug_dataloader, args, augment_strategy='random_exposed', recommendation_model=None):
        self.model = model
        self.dataloader = aug_dataloader
        self.args = args
        self.augment_data = []
        self.cuda_condition = check_gpu_capability() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.augment_strategy = augment_strategy
        if augment_strategy == 'bootstrap':
            self.recommender = recommendation_model
        elif augment_strategy == 'random_exposed':
            self.recommender = None
        else:
            raise NotImplementedError

    def bootstrap_augment(self):
        rec_data_iter = tqdm.tqdm(enumerate(self.dataloader),
                                  desc="Recommendation sequence augment",
                                  total=len(self.dataloader),
                                  bar_format="{l_bar}{r_bar}")
        self.model.eval()
        for i, batch in rec_data_iter:
            # batch = tuple(t.to(self.device) for t in batch)
            sequence, actions, states, timesteps, seq_len, behaviours = batch
            """
            BATCH:
            sequence [(item0, behaviour0),...,(item18, behaviour18)] should cat with new items
            actions [item10,item11,...,item18,item19*,...,item28*] fixed  [B,cl+aug_length-1,1]
            states [[item0,item1,...,item9],...,[item18,item19*,...,item27*]] fixed [B,cl+aug_length-1,el]
            timesteps [10,11,12,...19] fixed [B,aug_length,1] 
            seq_len [10,10,10,...,10] fixed [B,cl+aug_length-1]
            behaviours [[beh0,beh1,...,beh9],...,[beh9,...,beh18]] [B,cl,el] should cat with new behaviours

            reward model's input should be:
            # states: (batch, context_len, gru_len)
            # behaviours: (batch, context_len, gru_len)
            # actions: (batch, context_len, 1)
            # timesteps: (batch, 1, 1)
            # seq_len: (batch, context_len)
            
            recommender model's input should be:
            # states: (batch, context_len, gru_len)
            # behaviours: (batch, context_len, gru_len)
            # actions: (batch, context_len, 1)
            # rtgs: (batch, context_len, 1)
            # timesteps: (batch, 1, 1)
            # seq_len: (batch, context_len)
            """
            actions = actions[:, :self.args.context_length - 1, :].to(self.device)
            rewards = sequence[:, -(self.args.context_length - 1):, -1].to(self.device)
            rtgs = torch.zeros_like(rewards).to(self.device)
            for i in range(rewards.size(0)):
                rtgs[i] = rewards[i].flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,))
            rtgs += 1
            column = torch.full((rewards.size(0), 1), 1).to(self.device)
            rtgs = torch.cat((rtgs, column), dim=1).unsqueeze(-1).to(self.device)
            rtgs += self.args.aug_length
            states = states[:, :self.args.context_length, :].to(self.device)
            behaviours = behaviours[:, :self.args.context_length, :].to(self.device)
            for index in range(self.args.aug_length):
                # obtain input_timesteps & input_seq_len from original AugDataset
                input_timesteps = timesteps[:, index, :].unsqueeze(-1).to(self.device)
                input_seq_len = seq_len[:, index:index + self.args.context_length].to(self.device)

                # obtain others dynamically
                input_actions = actions[:, -(self.args.context_length - 1):, :]
                input_states = states[:, -self.args.context_length:, :]
                input_behaviours = behaviours[:, -self.args.context_length:, :]
                input_rtgs = rtgs[:, -self.args.context_length:, :]
                logits = self.recommender.predict(states=input_states, actions=input_actions,
                                                  behaviours=input_behaviours, seq_len=input_seq_len, rtgs=input_rtgs,
                                                  timesteps=input_timesteps)
                predict_actions = logits.argmax(dim=1).unsqueeze(1)  # (batch,1)
                actions = torch.cat((actions, predict_actions.unsqueeze(-1)), dim=1)
                input_actions = actions[:, -self.args.context_length:, :]
                predict_rewards = self.model.predict(states=input_states, actions=input_actions, seq_len=input_seq_len,
                                                     behaviours=input_behaviours,
                                                     timesteps=input_timesteps)
                # print(f'predict_actions={predict_actions.shape}') # predict_actions=torch.Size([64, 1])
                # print(f'predict_rewards={predict_rewards.shape}') # predict_rewards=torch.Size([64])
                predict_item = torch.cat(
                    (predict_actions.cpu(), predict_rewards.cpu().unsqueeze(-1)),
                    dim=-1)
                sequence = torch.cat((sequence, predict_item.unsqueeze(1)), dim=1)

                last_behaviours = behaviours[:, -1, :].squeeze()
                new_behaviours = torch.cat((last_behaviours, predict_rewards.unsqueeze(-1)), dim=-1)[:,
                                 -self.args.encoder_length:]
                behaviours = torch.cat((behaviours, new_behaviours.unsqueeze(1)), dim=1)

                last_states = states[:, -1, :].squeeze()
                # print(f'last_states={last_states.shape}')  # predict_actions=torch.Size([64, 1])
                # print(f'predict_actions={predict_actions.shape}')  # predict_rewards=torch.Size([64])
                new_states = torch.cat((last_states, predict_actions), dim=-1)[:,
                             -self.args.encoder_length:]
                states = torch.cat((states, new_states.unsqueeze(1)), dim=1)

                last_rtgs = rtgs[:, -1, :].squeeze()
                new_rtgs = last_rtgs - predict_rewards
                rtgs = torch.cat((rtgs, new_rtgs.unsqueeze(1).unsqueeze(2)), dim=1)

            self.augment_data.extend(sequence.tolist())

    def augment(self):
        if self.augment_strategy == 'bootstrap':
            self.bootstrap_augment()
        elif self.augment_strategy == 'random_exposed':
            self.random_augment()
        else:
            raise NotImplementedError

    def random_augment(self):
        rec_data_iter = tqdm.tqdm(enumerate(self.dataloader),
                                  desc="Recommendation sequence augment",
                                  total=len(self.dataloader),
                                  bar_format="{l_bar}{r_bar}")
        self.model.eval()
        for i, batch in rec_data_iter:
            # batch = tuple(t.to(self.device) for t in batch)
            sequence, actions, states, timesteps, seq_len, behaviours = batch
            """
            BATCH:
            sequence [(item0, behaviour0),...,(item18, behaviour18)] should cat with new items
            actions [item10,item11,...,item18,item19*,...,item28*] fixed  [B,cl+aug_length-1,1]
            states [[item0,item1,...,item9],...,[item18,item19*,...,item27*]] fixed [B,cl+aug_length-1,el]
            timesteps [10,11,12,...19] fixed [B,aug_length,1] 
            seq_len [10,10,10,...,10] fixed [B,cl+aug_length-1]
            behaviours [[beh0,beh1,...,beh9],...,[beh9,...,beh18]] [B,cl,el] should cat with new behaviours
            
            INPUT should be:
            # states: (batch, context_len, gru_len)
            # behaviours: (batch, context_len, gru_len)
            # actions: (batch, context_len, 1)
            # timesteps: (batch, 1, 1)
            # seq_len: (batch, context_len)
            """
            for index in range(self.args.aug_length):
                input_actions = actions[:, index:index + self.args.context_length, :]
                predict_actions = input_actions[:, -1, :].squeeze()
                input_actions = input_actions.to(self.device)
                input_states = states[:, index:index + self.args.context_length, :].to(self.device)
                input_timesteps = timesteps[:, index, :].unsqueeze(-1).to(self.device)
                input_seq_len = seq_len[:, index:index + self.args.context_length].to(self.device)
                input_behaviours = behaviours[:, -self.args.context_length:, :].to(self.device)
                predict_rewards = self.model.predict(states=input_states, actions=input_actions, seq_len=input_seq_len,
                                                     behaviours=input_behaviours,
                                                     timesteps=input_timesteps).cpu()
                # print(f'predict_item={predict_actions.shape}')
                # print(f'sequence={predict_rewards.shape}')
                predict_item = torch.cat((predict_actions.unsqueeze(-1), predict_rewards.unsqueeze(-1)), dim=-1)
                # print(f'predict_item={predict_item.shape}')
                # print(f'sequence={sequence.shape}')
                sequence = torch.cat((sequence, predict_item.unsqueeze(1)), dim=1)
                last_behaviours = behaviours[:, -1, :].squeeze()
                # print(f'last_behaviours={last_behaviours.shape}')
                # print(f'predict_rewards={predict_rewards.shape}')
                new_behaviours = torch.cat((last_behaviours, predict_rewards.unsqueeze(-1)), dim=-1)[:,
                                 -self.args.encoder_length:]
                behaviours = torch.cat((behaviours, new_behaviours.unsqueeze(1)), dim=1)

            self.augment_data.extend(sequence.tolist())

    def save_data(self, path):
        def augment(seq_list):
            u2seq = []
            for seq in seq_list:
                for i in range(2, len(seq) + 1):
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

        def tuple2file(data, dfile):
            with open(dfile, 'w') as f:
                append_items = set()
                for items in data:
                    items = [tuple(item) for item in items]
                    items = truncate_list(items)
                    if items is None:
                        continue
                    if tuple(items) not in append_items:
                        append_items.add(tuple(items))
                        for item in items:
                            f.write(str(item[0]) + ':' + str(item[1]) + " ")
                        f.write('\n')

        save_data = augment(self.augment_data)
        tuple2file(save_data, path)


if __name__ == '__main__':
    ori_seq = []
    for i in range(19):
        ori_seq.append((i, i % 2))
    seq_tensor = torch.tensor(ori_seq, dtype=torch.long)
    print(seq_tensor.shape)
    rewards = seq_tensor[-9:, -1]
    print(rewards)
    rewards = rewards.unsqueeze(0).expand(64, -1)
    result = torch.zeros_like(rewards)
    for i in range(rewards.size(0)):
        result[i] = rewards[i].flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,))
    print(result)
