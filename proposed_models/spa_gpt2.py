import os
import torch
import torch.nn as nn

from utils.model_io import capture_init_args

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from .components.embed import DataEmbedding

from .components.channel_attention import MultiSpectralAttentionLayer

# Mirror of the Hugging Face model hub to avoid the error of downloading the model
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co' # This is the default endpoint


class SPAGPT2(nn.Module):

    def __init__(self, 
        feature_size:int,
        seq_len=24, 
        pred_len=6,
        gpt2_type='gpt2',
        gpt_layers=6, 
        mlp=False,
        embed_size = 64,
        conv_layers = 4,
        conv_dim = 128,  
        hidden_size = 128,
        dropout=0.1
    ):
        super(SPAGPT2, self).__init__()
        capture_init_args(self, locals())   # for reproducibility, save __init__ args in instance._init_args
        self.feature_size = feature_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.gpt2_type = gpt2_type
        self.gpt_layers = gpt_layers
        self.mlp = mlp  # indicates whether to re-train the mlp layers of gpt2
        
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # before Spectral Attention 
        self.embed_size = embed_size
        self.conv1d_emb = nn.Sequential(
            nn.Conv1d(self.feature_size, self.hidden_size, 1),  
            nn.Conv1d(self.hidden_size, self.feature_size*self.embed_size, 1)
        )

        # Spectral Attention
        self.conv_layers = conv_layers
        self.conv_dim = conv_dim
        self.sp_attention = nn.Sequential(nn.Conv2d(self.seq_len, self.conv_dim, 3, 1, 1))
        for i in range(self.conv_layers):
            self.sp_attention.append(
                MultiSpectralAttentionLayer(
                    channel=self.conv_dim,
                    dct_h=7,
                    dct_w=7,
                    reduction=16,
                    freq_sel_method='top32'
                )
            )
        self.sp_attention.append(nn.Conv2d(self.conv_dim, self.seq_len, 3, 1, 1))


        # Load the GPT2 model with the specified type and number of layers
        if self.gpt2_type == 'gpt2-medium':
            self.gpt2 = GPT2Model.from_pretrained('gpt2-medium', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:self.gpt_layers]
            self.gpt_dim = 1024
        elif self.gpt2_type == 'gpt2-large':
            self.gpt2 = GPT2Model.from_pretrained('gpt2-large', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:self.gpt_layers]
            self.gpt_dim = 1280
        elif self.gpt2_type == 'gpt2-xl':
            self.gpt2 = GPT2Model.from_pretrained('gpt2-xl', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:self.gpt_layers]
            self.gpt_dim = 1600
        else:
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:self.gpt_layers]
            self.gpt_dim = 768

        # Set the parameters that require gradient for training
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            elif 'mlp' in name and self.mlp is True:
                param.requires_grad = True
                print(f"Use MLP layers for training")
            else:
                param.requires_grad = False
        
        # Embedding layer for gpt2 input
        self.gpt2_embedding = DataEmbedding(self.feature_size*self.embed_size, self.gpt_dim, self.dropout)
        
        # Linear layer for output predictions for gpt2 output last hidden states
        self.output_layer_feature = nn.Sequential(
            nn.Conv1d(self.gpt_dim, self.hidden_size, 1),
            nn.Conv1d(self.hidden_size, self.feature_size, 1)
        )
        self.output_layer_time = nn.Sequential(
            nn.Conv1d(self.seq_len, self.hidden_size, 1),
            nn.Conv1d(self.hidden_size, self.pred_len, 1)
        )
        

    def forward(self, x_enc):
        # normalize the input data
        mean = torch.mean(x_enc)
        std = torch.std(x_enc)
        x_enc = (x_enc - mean) / std
        
        B, L, N = x_enc.shape
        
        x_enc = self.conv1d_emb(x_enc.permute(0, 2, 1)).permute(0, 2, 1) # B, L, N*D
        x_enc = x_enc.reshape(B, L, N, self.embed_size)
        x_enc = self.sp_attention(x_enc) # B, L, N, D
        x_enc = x_enc.reshape(B, L, N*self.embed_size)

        # embedding layer
        x_embed = self.gpt2_embedding(x_enc)  # [B, L, 768]

        # gpt2 model
        x_dec = self.gpt2(inputs_embeds=x_embed).last_hidden_state  # [B, L, 768]

        # output layer
        x_dec = self.output_layer_feature(x_dec.permute(0, 2, 1)).permute(0, 2, 1)
        x_dec = self.output_layer_time(x_dec)

        # denormalize the output data
        x_dec = x_dec * std + mean

        return x_dec[:, -self.pred_len:, :]  # [B, L, N]
    
    
    
class SPAGPT2_wo_GPT2(nn.Module):

    def __init__(self, 
        feature_size:int,
        seq_len=24, 
        pred_len=6,
        gpt2_type='gpt2',
        gpt_layers=6, 
        mlp=False,
        embed_size = 64,
        conv_layers = 4,
        conv_dim = 128,  
        hidden_size = 128,
        dropout=0.1
    ):
        super(SPAGPT2_wo_GPT2, self).__init__()
        capture_init_args(self, locals())   # for reproducibility, save __init__ args in instance._init_args
        self.feature_size = feature_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.gpt2_type = gpt2_type
        self.gpt_layers = gpt_layers
        self.mlp = mlp  # indicates whether to re-train the mlp layers of gpt2
        
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # before Spectral Attention 
        self.embed_size = embed_size
        self.conv1d_emb = nn.Sequential(
            nn.Conv1d(self.feature_size, self.hidden_size, 1),  
            nn.Conv1d(self.hidden_size, self.feature_size*self.embed_size, 1)
        )

        # Spectral Attention
        self.conv_layers = conv_layers
        self.conv_dim = conv_dim
        self.sp_attention = nn.Sequential(nn.Conv2d(self.seq_len, self.conv_dim, 3, 1, 1))
        for i in range(self.conv_layers):
            self.sp_attention.append(
                MultiSpectralAttentionLayer(
                    channel=self.conv_dim,
                    dct_h=7,
                    dct_w=7,
                    reduction=16,
                    freq_sel_method='top32'
                )
            )
        self.sp_attention.append(nn.Conv2d(self.conv_dim, self.seq_len, 3, 1, 1))

        self.gpt_dim = 768
        # Load the GPT2 model with the specified type and number of layers
        # if self.gpt2_type == 'gpt2-medium':
        #     self.gpt2 = GPT2Model.from_pretrained('gpt2-medium', output_attentions=True, output_hidden_states=True)
        #     self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        #     self.gpt_dim = 1024
        # elif self.gpt2_type == 'gpt2-large':
        #     self.gpt2 = GPT2Model.from_pretrained('gpt2-large', output_attentions=True, output_hidden_states=True)
        #     self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        #     self.gpt_dim = 1280
        # elif self.gpt2_type == 'gpt2-xl':
        #     self.gpt2 = GPT2Model.from_pretrained('gpt2-xl', output_attentions=True, output_hidden_states=True)
        #     self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        #     self.gpt_dim = 1600
        # else:
        #     self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        #     self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        #     self.gpt_dim = 768

        # Set the parameters that require gradient for training
        # for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #     if 'ln' in name or 'wpe' in name:
        #         param.requires_grad = True
        #     elif 'mlp' in name and self.mlp is True:
        #         param.requires_grad = True
        #         print(f"Use MLP layers for training")
        #     else:
        #         param.requires_grad = False
        
        # Embedding layer for gpt2 input
        self.gpt2_embedding = DataEmbedding(self.feature_size*self.embed_size, self.gpt_dim, self.dropout)
        
        # Linear layer for output predictions for gpt2 output last hidden states
        self.output_layer_feature = nn.Sequential(
            nn.Conv1d(self.gpt_dim, self.hidden_size, 1),
            nn.Conv1d(self.hidden_size, self.feature_size, 1)
        )
        self.output_layer_time = nn.Sequential(
            nn.Conv1d(self.seq_len, self.hidden_size, 1),
            nn.Conv1d(self.hidden_size, self.pred_len, 1)
        )
        

    def forward(self, x_enc):
        # normalize the input data
        mean = torch.mean(x_enc)
        std = torch.std(x_enc)
        x_enc = (x_enc - mean) / std
        
        B, L, N = x_enc.shape
        
        x_enc = self.conv1d_emb(x_enc.permute(0, 2, 1)).permute(0, 2, 1) # B, L, N*D
        x_enc = x_enc.reshape(B, L, N, self.embed_size)
        x_enc = self.sp_attention(x_enc) # B, L, N, D
        x_enc = x_enc.reshape(B, L, N*self.embed_size)

        # embedding layer
        x_embed = self.gpt2_embedding(x_enc)  # [B, L, 768]
        x_dec = x_embed
        # # gpt2 model
        # x_dec = self.gpt2(inputs_embeds=x_embed).last_hidden_state  # [B, L, 768]

        # output layer
        x_dec = self.output_layer_feature(x_dec.permute(0, 2, 1)).permute(0, 2, 1)
        x_dec = self.output_layer_time(x_dec)

        # denormalize the output data
        x_dec = x_dec * std + mean

        return x_dec[:, -self.pred_len:, :]  # [B, L, N]
    
        
        
        