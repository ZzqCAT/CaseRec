"""
@Time       : 2024/6/6 14:41
@File       : model_DT.py
@Description: DT4REC
"""

"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

logger = logging.getLogger(__name__)

import numpy as np


class Seq2SeqEncoder(nn.Module):
    """用于序列到序列学习的循环神经网络编码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0., behaviour_size=2, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout, batch_first=True)
        self.behaviour_embedding = nn.Embedding(behaviour_size, embed_size)

    def forward(self, X, seq_len, behaviours, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        # print(f'X={X.device}')
        X = self.embedding(X)
        behaviour_embd = self.behaviour_embedding(behaviours)
        X = X + behaviour_embd
        # if seq_len==0, which means in this step s, a, r are none,
        # states could be random as the loss will not be calculated.
        seq_len[seq_len == 0] = 1
        seq_len = seq_len.to('cpu')
        packed_embedded = pack_padded_sequence(X, lengths=seq_len, batch_first=True, enforce_sorted=False)
        output, state = self.rnn(packed_embedded)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                             .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.state_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, 2, 0.1,
                                            config.behaviour_size)

        self.rtg_embeddings = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        if not self.config.share_encoder:
            self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        else:
            self.action_embeddings = nn.Sequential(self.state_encoder.embedding, nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if not self.config.share_para4decoder:
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # else:
        #     embedding_weights = self.action_embeddings[0].weight
        #     # Create a new linear layer
        #     self.head = nn.Linear(embedding_weights.size(1), embedding_weights.size(0), bias=False)
        #     # Initialize the weight of the linear layer with the embedding weights (transposed)
        #     self.head.weight = nn.Parameter(embedding_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if 'bias' in pn:
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif 'weight' in pn and isinstance(m, torch.nn.GRU):
                    decay.add(fpn)
                # else:
                #     decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, seq_len, behaviours, targets=None, rtgs=None, timesteps=None):
        # states: (batch, context_len, gru_len)
        # behaviours: (batch, context_len, gru_len)
        # actions: (batch, context_len, 1)
        # targets: (batch, context_len, 1)
        # rtgs: (batch, context_len, 1)
        # timesteps: (batch, 1, 1)
        # seq_len: (batch, context_len)
        device = states.device
        state_embeddings = torch.zeros([states.shape[0], states.shape[1], self.config.n_embd]).to(device)
        seq_len = seq_len.reshape(-1, states.shape[0])  # context_len,batch_size
        for i in range(states.shape[1]):
            states_seq = states[:, i, :].type(torch.long).squeeze()
            behaviours_seq = behaviours[:, i, :].type(torch.long).squeeze()
            # print(f'states_seq={states_seq.shape}')
            output, state = self.state_encoder(states_seq, seq_len[i], behaviours_seq)
            # print(f'state={state.shape}')
            # print(f'state_embeddings={state_embeddings.shape}')
            context = state.permute(1, 0, 2)
            # context = (batch_size, num_layers, hidden_size) get last layer hidden states
            state_embeddings[:, i, :] = context[:, -1, :]

        if actions is not None and self.model_type == 'reward_conditioned':
            rtg_embeddings = self.rtg_embeddings(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 3 - int(targets is None), self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]
        # this will not happen because action is padding with 0
        elif actions is None and self.model_type == 'reward_conditioned':
            rtg_embeddings = self.rtg_embeddings(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1] * 2, self.config.n_embd),
                                           dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, ::2, :] = rtg_embeddings  # really just [:,0,:]
            token_embeddings[:, 1::2, :] = state_embeddings  # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2 - int(targets is None), self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]
        elif actions is None and self.model_type == 'naive':  # only happens at very first timestep of training
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        token_embeddings = token_embeddings.to(device)

        # token_embeddings (B, block_size, embd_size)
        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size,
                                                     dim=0)  # batch_size, traj_length, n_embd

        position_embeddings = \
            torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) \
            + self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        if not self.config.share_para4decoder:
            logits = self.head(x)
        else:
            embedding_weights = self.action_embeddings[0].weight
            logits = F.linear(x, embedding_weights)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
        # keep the logits size the same for calculating loss
        # elif actions is None and self.model_type == 'reward_conditioned':
        #     logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits  # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=0) / self.config.context_length

        return logits, loss

    def predict(self, states, actions, seq_len, behaviours, rtgs=None, timesteps=None, temperature=1.0):
        logits, _ = self.forward(states, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps, seq_len=seq_len,
                                 behaviours=behaviours)
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        return logits.squeeze()


class RewardTransformer(nn.Module):
    """  Reward model, give state & action => reward (0/1) """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.state_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, 2, 0.1,
                                            config.behaviour_size)
        if not self.config.share_encoder:
            self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        else:
            self.action_embeddings = nn.Sequential(self.state_encoder.embedding, nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, 2, bias=False)
        # else:
        #     embedding_weights = self.action_embeddings[0].weight
        #     # Create a new linear layer
        #     self.head = nn.Linear(embedding_weights.size(1), embedding_weights.size(0), bias=False)
        #     # Initialize the weight of the linear layer with the embedding weights (transposed)
        #     self.head.weight = nn.Parameter(embedding_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if 'bias' in pn:
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif 'weight' in pn and isinstance(m, torch.nn.GRU):
                    decay.add(fpn)
                # else:
                #     decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, seq_len, behaviours, targets=None, timesteps=None):
        # states: (batch, context_len, gru_len)
        # behaviours: (batch, context_len, gru_len)
        # actions: (batch, context_len, 1)
        # targets: (batch, context_len, 1)
        # timesteps: (batch, 1, 1)
        # seq_len: (batch, context_len)
        device = states.device
        state_embeddings = torch.zeros([states.shape[0], states.shape[1], self.config.n_embd]).to(device)
        seq_len = seq_len.reshape(-1, states.shape[0])  # context_len,batch_size
        for i in range(states.shape[1]):
            states_seq = states[:, i, :].type(torch.long).squeeze()
            behaviours_seq = behaviours[:, i, :].type(torch.long).squeeze()
            # print(f'states_seq={states_seq.shape}')
            # print(f'states_seq={states_seq.device}')
            # print(f'seq_len[i]={seq_len[i].device}')
            # print(f'behaviours_seq={behaviours_seq.device}')
            output, state = self.state_encoder(states_seq, seq_len[i], behaviours_seq)
            # print(f'state={state.shape}')
            # print(f'state_embeddings={state_embeddings.shape}')
            context = state.permute(1, 0, 2)
            # context = (batch_size, num_layers, hidden_size) get last layer hidden states
            state_embeddings[:, i, :] = context[:, -1, :]

        if actions is not None and self.model_type == 'reward_conditioned':
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2 - int(targets is None), self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]
        # this will not happen because action is padding with 0
        elif actions is None and self.model_type == 'reward_conditioned':

            token_embeddings = torch.zeros((states.shape[0], states.shape[1] * 1, self.config.n_embd),
                                           dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, 1::, :] = state_embeddings  # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2 - int(targets is None), self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]
        elif actions is None and self.model_type == 'naive':  # only happens at very first timestep of training
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        token_embeddings = token_embeddings.to(device)

        # token_embeddings (B, block_size, embd_size)
        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size,
                                                     dim=0)  # batch_size, traj_length, n_embd

        position_embeddings = \
            torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) \
            + self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::2, :]  # only keep predictions from state_embeddings
        # keep the logits size the same for calculating loss
        # elif actions is None and self.model_type == 'reward_conditioned':
        #     logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits  # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-1)

        return logits, loss

    def predict(self, states, actions, seq_len, behaviours, timesteps=None, temperature=1.0, prob_depend=False,
                threshold=0.5):
        logits, _ = self.forward(states, actions=actions, targets=None, timesteps=timesteps, seq_len=seq_len,
                                 behaviours=behaviours)
        logits = logits[:, -1, :] / temperature
        # logits (batch,2)
        probs = F.softmax(logits, dim=-1)
        if prob_depend:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            if threshold != 0.5:
                ix = (probs[:, 1] > threshold).long().unsqueeze(1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
        # ix (batch,1)
        return ix.squeeze()


class VanillaGPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # self.state_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, 2, 0.1)

        self.rtg_embeddings = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())

        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if not self.config.share_para4decoder:
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # else:
        #     embedding_weights = self.action_embeddings[0].weight
        #     # Create a new linear layer
        #     self.head = nn.Linear(embedding_weights.size(1), embedding_weights.size(0), bias=False)
        #     # Initialize the weight of the linear layer with the embedding weights (transposed)
        #     self.head.weight = nn.Parameter(embedding_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if 'bias' in pn:
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif 'weight' in pn and isinstance(m, torch.nn.GRU):
                    decay.add(fpn)
                # else:
                #     decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, seq_len, behaviours, targets=None, rtgs=None, timesteps=None):
        # states: (batch, context_len, gru_len)
        # actions: (batch, context_len, 1)
        # targets: (batch, context_len, 1)
        # rtgs: (batch, context_len, 1)
        # timesteps: (batch, 1, 1)
        # seq_len: (batch, context_len)
        device = states.device

        # state_embeddings = torch.zeros([states.shape[0], states.shape[1], self.config.n_embd]).to(device)
        # embedding_layer = self.action_embeddings[0]
        # state_embeddings = embedding_layer(states[:, :, 0].type(torch.long).squeeze(-1))
        # rtgs = torch.ones_like(rtgs).to(device)
        # 遍历 batch 和 context_len 维度

        # for c in range(states.shape[1]):
        #     embeddings = embedding_layer(states[:, c, :])
        #     # print(embeddings.shape)
        #     # b = embeddings.shape[0]
        #     # e = embeddings.shape[2]
        #     # # embeddings b,gru_len,e
        #     # indices = torch.arange(10).unsqueeze(0).expand(states.shape[0], -1).to(device)
        #     # print(indices.shape)
        #     # valid_indices = indices < seq_len[:, c].unsqueeze(1).to(device)
        #     # valid_states = embeddings[valid_indices]
        #     # print(valid_states.shape)
        #     # valid_states = valid_states.view(b, -1, e).mean(dim=1).to(device)
        #     state_embeddings[:, c, :] = torch.mean(embeddings, dim=1)

        # for b in range(states.shape[0]):
        #     valid_length = seq_len[b, c]  # 获取当前 batch 的长度
        #     state_embeddings[b, c, :] = embeddings[b, :valid_length].mean(dim=0)

        # for b in range(states.shape[0]):
        #     for c in range(states.shape[1]):
        #         # 根据 seq_len[b, c] 截取 states[b, c] 的前 seq_len[b, c] 个元素
        #         valid_length = seq_len[b, c]
        #         if valid_length > 0:
        #             # 获取对应的前 valid_length 个 token 的索引
        #             valid_states = states[b, c, :valid_length]  # 转为 long 型以传递给 embedding
        #
        #             # 通过 nn.Embedding 获取嵌入
        #             embeddings = embedding_layer(valid_states)
        #
        #             # 对这些嵌入取平均值
        #             avg_embedding = embeddings.mean(dim=0)
        #
        #             # 存放到 output 中
        #             state_embeddings[b, c, :] = avg_embedding

        state_embeddings = torch.zeros([states.shape[0], states.shape[1], self.config.n_embd]).to(device)
        seq_len = seq_len.reshape(-1, states.shape[0])  # context_len,batch_size
        for i in range(states.shape[1]):
            states_seq = states[:, i, :].type(torch.long).squeeze()
            behaviours_seq = behaviours[:, i, :].type(torch.long).squeeze()
            # print(f'states_seq={states_seq.shape}')
            output, state = self.state_encoder(states_seq, seq_len[i], behaviours_seq)
            # print(f'state={state.shape}')
            # print(f'state_embeddings={state_embeddings.shape}')
            context = state.permute(1, 0, 2)
            # context = (batch_size, num_layers, hidden_size) get last layer hidden states
            state_embeddings[:, i, :] = context[:, -1, :]

        if actions is not None and self.model_type == 'reward_conditioned':
            rtg_embeddings = self.rtg_embeddings(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 3 - int(targets is None), self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]
        # this will not happen because action is padding with 0
        elif actions is None and self.model_type == 'reward_conditioned':
            rtg_embeddings = self.rtg_embeddings(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1] * 2, self.config.n_embd),
                                           dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, ::2, :] = rtg_embeddings  # really just [:,0,:]
            token_embeddings[:, 1::2, :] = state_embeddings  # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2 - int(targets is None), self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]
        elif actions is None and self.model_type == 'naive':  # only happens at very first timestep of training
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        token_embeddings = token_embeddings.to(device)

        # token_embeddings (B, block_size, embd_size)
        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size,
                                                     dim=0)  # batch_size, traj_length, n_embd

        position_embeddings = \
            torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) \
            + self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        if not self.config.share_para4decoder:
            logits = self.head(x)
        else:
            embedding_weights = self.action_embeddings[0].weight
            logits = F.linear(x, embedding_weights)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
        # keep the logits size the same for calculating loss
        # elif actions is None and self.model_type == 'reward_conditioned':
        #     logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits  # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits[:-1].reshape(-1, logits.size(-1)), targets[:-1].reshape(-1),
                                   ignore_index=0) / self.config.context_length

        return logits, loss

    def predict(self, states, actions, seq_len, behaviours, rtgs=None, timesteps=None, temperature=1.0):
        logits, _ = self.forward(states, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps, seq_len=seq_len,
                                 behaviours=behaviours)
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        return logits.squeeze()
