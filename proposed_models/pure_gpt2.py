import os
import torch
import torch.nn as nn

from utils.model_io import capture_init_args

from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from .components.embed import DataEmbedding

# Mirror of the Hugging Face model hub to avoid the error of downloading the model
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co' # This is the default endpoint


class GPT2(nn.Module):

    def __init__(self, 
        features_size:int,
        seq_len=24, 
        pred_len=6,
        gpt2_type='gpt2',
        gpt_layers=6, 
        mlp=True,  
        dropout=0.1
    ):
        super(GPT2, self).__init__()
        capture_init_args(self, locals())   # for reproducibility, save __init__ args in instance._init_args
        self.features_size = features_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.gpt2_type = gpt2_type
        self.gpt_layers = gpt_layers
        self.mlp = mlp  # indicates whether to re-train the mlp layers of gpt2
        
        self.dropout = dropout


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
            else:
                param.requires_grad = True
        
        # Embedding layer for gpt2 input
        self.gpt2_embedding = DataEmbedding(self.features_size, self.gpt_dim, self.dropout)
        
        # Linear layer for output predictions for gpt2 output last hidden states
        self.output_layer_feature = nn.Linear(self.gpt_dim, self.features_size)
        self.output_layer_time = nn.Linear(self.seq_len, self.pred_len)
        

    def forward(self, x_enc):
        # normalize the input data
        mean = torch.mean(x_enc)
        std = torch.std(x_enc)
        x_enc = (x_enc - mean) / std
        
        B, L, N = x_enc.shape

        # embedding layer
        x_embed = self.gpt2_embedding(x_enc)  # [B, L, 768]

        # gpt2 model
        x_dec = self.gpt2(inputs_embeds=x_embed).last_hidden_state  # [B, L, 768]

        # output layer
        x_dec = self.output_layer_feature(x_dec)
        x_dec = self.output_layer_time(x_dec.permute(0, 2, 1)).permute(0, 2, 1)

        # denormalize the output data
        x_dec = x_dec * std + mean

        return x_dec[:, -self.pred_len:, :]  # [B, L, N]
    
        
        
        