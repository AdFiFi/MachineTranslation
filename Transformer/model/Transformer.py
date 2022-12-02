import torch
from torch import nn

from .Embed import DataEmbedding

from .Transformer_EncDec import Encoder, Decoder


class TransformerConfig:
    def __init__(self, enc_vocab_size, dec_vocab_size, d_model=512, num_heads=8, dim_feedforward=512, batch_size=128,
                 num_encoder_layers=3, num_decoder_layers=3, activation='gelu', dropout=0.1, output_attention=False):
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.batch_size = batch_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.activation = activation
        self.dropout = dropout
        self.output_attention = output_attention


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.output_attention = config.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(config.enc_vocab_size, config.d_model, config.dropout)
        self.dec_embedding = DataEmbedding(config.dec_vocab_size, config.d_model, config.dropout)
        # Encoder
        self.encoder = Encoder(config)
        # Decoder
        self.decoder = Decoder(config)

    def forward(self, enc_ids, dec_ids, enc_padding_mask, dec_padding_mask,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_encoding = self.enc_embedding(enc_ids, enc_padding_mask)
        enc_encoding, attns = self.encoder(enc_encoding, attn_mask=enc_self_mask)

        dec_encoding = self.dec_embedding(dec_ids, dec_padding_mask)
        dec_out = self.decoder(dec_encoding, enc_encoding, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
