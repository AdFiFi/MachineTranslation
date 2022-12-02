import torch
from torch import nn

from .Embed import DataEmbedding
from .Attention_Family import FullAttention, AttentionLayer
from .Transformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer


class TransformerConfig:
    def __init__(self, enc_vocab_size, dec_vocab_size, emb_size=512, num_heads=8, dim_feedforward=512, batch_size=128,
                 num_encoder_layers=3, num_decoder_layers=3, activation='gelu'):
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.batch_size = batch_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.activation = activation


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.output_attention = config.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(config.enc_vocab_size, config.d_model, config.dropout)
        self.dec_embedding = DataEmbedding(config.dec_vocab_size, config.d_model, config.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=config.dropout, output_attention=config.output_attention),
                        config.d_model, config.num_heads),
                    config.d_model,
                    config.dim_feedforward,
                    dropout=config.dropout,
                    activation=config.activation
                ) for _ in range(config.num_encoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=config.dropout, output_attention=False),
                        config.d_model, config.num_heads),
                    AttentionLayer(
                        FullAttention(attention_dropout=config.dropout, output_attention=False),
                        config.d_model, config.num_heads),
                    config.d_model,
                    config.dim_feedforward,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.num_decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
            projection=nn.Linear(config.d_model, config.dec_vocab_size, bias=True)
        )

    def forward(self, enc_ids, enc_mark, dec_ids, dec_mark,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(enc_ids, enc_mark)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(dec_ids, dec_mark)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out
