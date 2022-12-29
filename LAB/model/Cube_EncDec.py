import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import TemporalAttention, SpatialAttention, TemporalAttentionLayer, SpatialAttentionLayer
from utils import triangular_causal_mask, temporal_mask_expand, spatial_mask_expand


class CubeEncoderLayer(nn.Module):
    def __init__(self, config):
        super(CubeEncoderLayer, self).__init__()
        self.temporal_attention = TemporalAttentionLayer(
            TemporalAttention(attention_dropout=config.dropout, output_attention=config.output_attention),
            config.d_model, config.num_t_heads)
        self.spatial_attention = SpatialAttentionLayer(
            SpatialAttention(attention_dropout=config.dropout, output_attention=config.output_attention),
            config.d_model, config.num_s_heads)
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = F.relu if config.activation == "relu" else F.gelu

    def forward(self, hidden_states, temporal_mask=None, spatial_mask=None):
        residual = hidden_states
        hidden_states, attn1 = self.temporal_attention(
            hidden_states, hidden_states, hidden_states,
            attn_mask=temporal_mask
        )
        hidden_states = residual + self.dropout(hidden_states)
        residual = hidden_states = self.norm1(hidden_states)

        hidden_states, attn2 = self.spatial_attention(
            hidden_states, hidden_states, hidden_states,
            attn_mask=spatial_mask
        )
        hidden_states = residual + self.dropout(hidden_states)
        residual = hidden_states = self.norm1(hidden_states)

        hidden_states = self.dropout(self.activation(self.linear1(hidden_states)))
        hidden_states = self.dropout(self.linear2(hidden_states))

        return self.norm2(residual + hidden_states), (attn1, attn2)


class CubeEncoder(nn.Module):
    def __init__(self, config):
        super(CubeEncoder, self).__init__()
        self.d_model = config.d_model
        self.attn_layers = nn.ModuleList([CubeEncoderLayer(config)
                                          for _ in range(config.num_encoder_layers)])
        self.norm = torch.nn.LayerNorm(config.d_model)

    def forward(self, enc_embeds, padding_mask=None, temporal_mask=None, spatial_mask=None):
        if padding_mask is not None:
            if temporal_mask is None:
                temporal_mask = temporal_mask_expand(padding_mask)
            if spatial_mask is None:
                # spatial_mask = spatial_mask_expand(padding_mask, self.d_model)
                spatial_mask = None

        # x [B, L, D]
        attns = []
        encoding = enc_embeds
        for attn_layer in self.attn_layers:
            encoding, attn = attn_layer(encoding, temporal_mask=temporal_mask, spatial_mask=spatial_mask)
            attns.append(attn)

        if self.norm is not None:
            encoding = self.norm(encoding)

        return encoding, attns


class CubeDecoderLayer(nn.Module):
    def __init__(self, config):
        super(CubeDecoderLayer, self).__init__()
        self.self_attention = TemporalAttentionLayer(
            TemporalAttention(attention_dropout=config.dropout, output_attention=False),
            config.d_model, config.num_t_heads)
        self.cross_attention = TemporalAttentionLayer(
            TemporalAttention(attention_dropout=config.dropout, output_attention=False),
            config.d_model, config.num_t_heads)
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


class CubeDecoder(nn.Module):
    def __init__(self, config):
        super(CubeDecoder, self).__init__()
        self.layers = nn.ModuleList([CubeDecoderLayer(config)
                                     for _ in range(config.num_decoder_layers)])
        self.norm = torch.nn.LayerNorm(config.d_model)
        self.projection = nn.Linear(config.d_model, config.dec_vocab_size, bias=True)

    def forward(self, dec_embeds, cross, padding_mask=None, attn_mask=None, cross_mask=None):
        if attn_mask is None:
            B, L, _ = dec_embeds.shape
            attn_mask = triangular_causal_mask(B, L, device=dec_embeds.device)
            if padding_mask is not None:
                attn_mask += temporal_mask_expand(padding_mask)
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
