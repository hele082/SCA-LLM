import os
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from einops import rearrange

from utils.model_io import capture_init_args

from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from .components.embed import DataEmbedding


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class ChannelAttention(nn.Module):
    
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Res_block(nn.Module):
    def __init__(self, in_planes):
        super(Res_block, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv1(x))
        rs1 = self.conv2(rs1)
        channel_attn = self.ca(rs1)
        output = channel_attn * rs1
        rs = torch.add(x, output)
        return rs


class LLM4CP(nn.Module):

    def __init__(self, 
        feature_size:int,
        seq_len=24, 
        pred_len=6,
        gpt_type='gpt2',   
        gpt_layers=6,
        mlp=False, 
        patch_size=4,
        res_layers=4,  
        res_dim=64,
        dropout=0.1
    ):
        super(LLM4CP, self).__init__()
        capture_init_args(self, locals())   # for reproducibility, save __init__ args in instance._init_args
        self.feature_size = feature_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.gpt_type = gpt_type
        self.gpt_layers = gpt_layers
        self.mlp = mlp
        if self.gpt_type == 'gpt2-medium':
            self.gpt2 = GPT2Model.from_pretrained('gpt2-medium', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:self.gpt_layers]
            self.gpt_dim = 1024
        elif self.gpt_type == 'gpt2-large':
            self.gpt2 = GPT2Model.from_pretrained('gpt2-large', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:self.gpt_layers]
            self.gpt_dim = 1280
        elif self.gpt_type == 'gpt2-xl':
            self.gpt2 = GPT2Model.from_pretrained('gpt2-xl', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:self.gpt_layers]
            self.gpt_dim = 1600
        else:
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:self.gpt_layers]
            self.gpt_dim = 768

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
                param.requires_grad = False
            elif 'mlp' in name and self.mlp is True:
                param.requires_grad = False
            else:
                param.requires_grad = False
               
        self.dropout = dropout
        self.gpt2_embedding = DataEmbedding(self.feature_size, self.gpt_dim, self.dropout)

        self.patch_size = patch_size
        assert self.seq_len % self.patch_size == 0, f"seq_len {self.seq_len} must be divisible by patch_size {self.patch_size}"
        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)
        self.patch_layer_fre = nn.Linear(self.patch_size, self.patch_size)
        self.predict_linear_pre = nn.Linear(self.seq_len, self.seq_len)
        
        # Linear layer for output predictions for gpt2 output last hidden states
        self.output_layer_feature = nn.Linear(self.gpt_dim, self.feature_size)
        self.output_layer_time = nn.Linear(self.seq_len, self.pred_len)

        # Residual Block for pre-processing data before embedding
        # parallel processing in delay and frequency domain
        self.res_layers = res_layers
        self.res_dim = res_dim
        self.RB_e = nn.Sequential(nn.Conv2d(2, self.res_dim, 3, 1, 1))
        self.RB_f = nn.Sequential(nn.Conv2d(2, self.res_dim, 3, 1, 1))
        for i in range(self.res_layers):
            self.RB_e.append(Res_block(self.res_dim))
            self.RB_f.append(Res_block(self.res_dim))
        self.RB_e.append(nn.Conv2d(self.res_dim, 2, 3, 1, 1))
        self.RB_f.append(nn.Conv2d(self.res_dim, 2, 3, 1, 1))

    def forward(self, x_enc):
        mean = torch.mean(x_enc)
        std = torch.std(x_enc)
        x_enc = (x_enc - mean) / std
        B, L, enc_in = x_enc.shape  # [B, L, D]
        
        
        # process in delay domain
        x_enc_r = rearrange(x_enc, 'b l (k o) -> b l k o', o=2)
        x_enc_complex = torch.complex(x_enc_r[:, :, :, 0], x_enc_r[:, :, :, 1])
        x_enc_delay = torch.fft.ifft(x_enc_complex, dim=2)
        x_enc_delay = torch.cat([torch.real(x_enc_delay), torch.imag(x_enc_delay)], dim=2)
        x_enc_delay = x_enc_delay.reshape(B, L // self.patch_size, self.patch_size, enc_in)
        x_enc_delay = self.patch_layer(x_enc_delay.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_delay = x_enc_delay.reshape(B, L, enc_in)
        x_enc_delay = rearrange(x_enc_delay, 'b l (k o) -> b o l k', o=2)
        x_enc_delay = self.RB_f(x_enc_delay)
        # process in frequency domain
        x_enc_fre = x_enc.reshape(B, L // self.patch_size, self.patch_size, enc_in)
        x_enc_fre = self.patch_layer_fre(x_enc_fre.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_fre = x_enc_fre.reshape(B, L, enc_in)
        x_enc_fre = rearrange(x_enc_fre, 'b l (k o) -> b o l k', o=2)
        x_enc_fre = self.RB_e(x_enc_fre)

        x_enc = x_enc_fre + x_enc_delay
        x_enc = rearrange(x_enc, 'b o l k -> b l (k o)', o=2)  # [B, L, D]

        enc_out = self.gpt2_embedding(x_enc)  # [B, L, 768]

        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        enc_out = torch.nn.functional.pad(enc_out, (0, self.gpt_dim - enc_out.shape[-1]))

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state  # [B, L, 768]

        dec_out = self.output_layer_feature(dec_out)
        dec_out = self.output_layer_time(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        

        dec_out = dec_out * std + mean

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

