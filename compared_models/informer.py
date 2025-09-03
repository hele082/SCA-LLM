import torch
import torch.nn as nn

from .components.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from .components.decoder import Decoder, DecoderLayer
from .components.attn import FullAttention, ProbAttention, AttentionLayer
from .components.embed import DataEmbedding

from utils.model_io import capture_init_args


class Informer(nn.Module):
    
    def __init__(self,
            n_features:int, 
            seq_len=24,  
            pred_len=6,
            label_len=10,
            factor=5, 
            d_model=64, 
            n_heads=8, 
            e_layers=4, 
            d_layers=3, 
            d_ff=64,
            dropout=0.1, 
            attn='full', 
            embed='fixed', 
            activation='gelu',
            distil=True,
            output_attention=False
        ):
        super(Informer, self).__init__()
        capture_init_args(self, locals())   # for reproducibility, save __init__ args in instance._init_args
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        
        enc_in = self.n_features
        dec_in = self.n_features
        c_out = self.n_features
        
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        

    def forward(self, 
                x_enc, 
                x_dec = None,
                enc_self_mask=None, 
                dec_self_mask=None, 
                dec_enc_mask=None):
        device = x_enc.device
        batch_size, seq_len, n_features = x_enc.shape
        
        if x_dec is None:
            # decoder input
            x_dec = torch.zeros_like(x_enc[:, -self.pred_len:, :]).float()
            x_dec = torch.cat([x_enc[:, -self.label_len:, :], x_dec], dim=1).float().to(device)
        
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


