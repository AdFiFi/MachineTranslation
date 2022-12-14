import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention_Family import FullAttention, AttentionLayer
from utils import triangular_causal_mask, mask_expand


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.attention = AttentionLayer(
            FullAttention(attention_dropout=config.dropout, output_attention=config.output_attention),
            config.d_model, config.num_heads)
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = F.relu if config.activation == "relu" else F.gelu

    def forward(self, hidden_states, attn_mask=None):
        residual = hidden_states
        hidden_states, attn = self.attention(
            hidden_states, hidden_states, hidden_states,
            attn_mask=attn_mask
        )
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states = self.norm1(hidden_states)
        hidden_states = self.dropout(self.activation(self.linear1(hidden_states)))
        hidden_states = self.dropout(self.linear2(hidden_states))

        return self.norm2(residual + hidden_states), attn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.norm = torch.nn.LayerNorm(config.d_model)

    def forward(self, enc_embeds, padding_mask=None, attn_mask=None):
        if padding_mask is not None:
            if attn_mask is None:
                attn_mask = mask_expand(padding_mask)

        # x [B, L, D]
        attns = []
        encoding = enc_embeds
        for attn_layer in self.attn_layers:
            encoding, attn = attn_layer(encoding,
                                        attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            encoding = self.norm(encoding)

        return encoding, attns


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attention = AttentionLayer(
            FullAttention(attention_dropout=config.dropout, output_attention=False),
            config.d_model, config.num_heads)
        self.cross_attention = AttentionLayer(
            FullAttention(attention_dropout=config.dropout, output_attention=False),
            config.d_model, config.num_heads)
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = F.relu if config.activation == "relu" else F.gelu

    def forward(self, hidden_states, cross, attn_mask=None, cross_mask=None):
        residual = hidden_states
        hidden_states = residual + self.dropout(self.self_attention(
            hidden_states, hidden_states, hidden_states,
            attn_mask=attn_mask)[0])
        residual = hidden_states = self.norm1(hidden_states)

        hidden_states = residual + self.dropout(self.cross_attention(
            hidden_states, cross, cross,
            attn_mask=cross_mask
        )[0])

        residual = hidden_states = self.norm2(hidden_states)
        hidden_states = self.dropout(self.activation(self.linear1(hidden_states)))
        hidden_states = self.dropout(self.linear2(hidden_states))

        return self.norm3(residual + hidden_states)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.norm = torch.nn.LayerNorm(config.d_model)
        self.projection = nn.Linear(config.d_model, config.dec_vocab_size, bias=True)

    def forward(self, dec_embeds, cross, padding_mask=None, attn_mask=None, cross_mask=None):
        if attn_mask is None:
            B, L, _ = dec_embeds.shape
            attn_mask = triangular_causal_mask(B, L, device=dec_embeds.device)
            if padding_mask is not None:
                attn_mask += mask_expand(padding_mask)
        decoding = dec_embeds

        for layer in self.layers:
            decoding = layer(decoding, cross,
                             attn_mask=attn_mask,
                             cross_mask=cross_mask)

        if self.norm is not None:
            decoding = self.norm(decoding)

        if self.projection is not None:
            decoding = self.projection(decoding)
        return decoding
