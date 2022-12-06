import torch
from torch import nn
# from transformers.generation_utils import GenerationMixin

from .Embed import DataEmbedding
from .Transformer_EncDec import Encoder, Decoder


class TransformerConfig:
    def __init__(self, enc_vocab_size=20000,
                 dec_vocab_size=20000,
                 max_seq_len=256,
                 d_model=512,
                 num_heads=8,
                 dim_feedforward=512,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 activation='gelu',
                 dropout=0.1,
                 output_attention=False,
                 pad_token_id=1,
                 bos_token_id=0,
                 eos_token_id=2,
                 ):
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.activation = activation
        self.dropout = dropout
        self.output_attention = output_attention
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.output_attention = config.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(config.enc_vocab_size, config.d_model, config.dropout)
        self.dec_embedding = DataEmbedding(config.dec_vocab_size, config.d_model, config.dropout)
        # Encoder
        self.encoder = Encoder(config)
        # Decoder
        self.decoder = Decoder(config)

        self.init_parameters()

    def forward(self, enc_ids, dec_ids, enc_padding_mask, dec_padding_mask,
                enc_attn_mask=None, dec_attn_mask=None, dec_enc_mask=None,
                use_cache=False, enc_encoding=None):
        attns = None
        if not use_cache and enc_encoding is None:
            enc_embeds = self.enc_embedding(enc_ids)
            enc_encoding, attns = self.encoder(enc_embeds,
                                               padding_mask=enc_padding_mask,
                                               attn_mask=enc_attn_mask)

        dec_embeds = self.dec_embedding(dec_ids)
        dec_out = self.decoder(dec_embeds, enc_encoding,
                               padding_mask=dec_padding_mask,
                               attn_mask=dec_attn_mask,
                               cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, enc_ids, enc_padding_mask, enc_attn_mask=None):
        enc_embeds = self.enc_embedding(enc_ids)
        enc_encoding, attns = self.encoder(enc_embeds,
                                           padding_mask=enc_padding_mask,
                                           attn_mask=enc_attn_mask)
        return enc_encoding, attns

    def decode(self, enc_encoding, dec_ids, past=None, **kwargs):
        dec_embeds = self.dec_embedding(dec_ids)
        dec_out = self.decoder(dec_embeds, enc_encoding, **kwargs)
        return dec_out

    def greedy_generate(self, enc_ids, enc_padding_mask, dec_ids, max_len=None, enc_attn_mask=None):
        enc_encoding, _ = self.encode(enc_ids, enc_padding_mask, enc_attn_mask)
        max_len = self.config.max_seq_len if max_len is None else min(max_len, self.config.max_seq_len)
        out_ids = None
        for i in range(max_len - 1):
            logits = self.decoder(enc_ids, dec_ids)
            out_ids = logits.data.max(2, keepdim=True)[1].squeeze(2)
            dec_ids = torch.cat((dec_ids[:, 0], out_ids), dim=1)

        return dec_ids, out_ids


#
# class TransformerForTranslation(Transformer, GenerationMixin):
#     def __init__(self, config: TransformerConfig):
#         super().__init__(config)
#
#     def forward(self, enc_ids, dec_ids, enc_padding_mask, dec_padding_mask,
#                 enc_attn_mask=None, dec_attn_mask=None, dec_enc_mask=None,
#                 past_key_values=None, use_cache=None, encoder_outputs=None):
#
#         enc_embeds = self.enc_embedding(enc_ids)
#         enc_encoding, attns = self.encoder(enc_embeds,
#                                            padding_mask=enc_padding_mask,
#                                            attn_mask=enc_attn_mask)
#
#         dec_embeds = self.dec_embedding(dec_ids)
#         dec_out = self.decoder(dec_embeds, enc_encoding,
#                                padding_mask=dec_padding_mask,
#                                attn_mask=dec_attn_mask,
#                                cross_mask=dec_enc_mask)
#
#         if self.output_attention:
#             return dec_out, attns
#         else:
#             return dec_out
#
#     def encode(self, enc_ids, enc_padding_mask, enc_attn_mask=None):
#         enc_embeds = self.enc_embedding(enc_ids)
#         enc_encoding, _ = self.encoder(enc_embeds,
#                                        padding_mask=enc_padding_mask,
#                                        attn_mask=enc_attn_mask)
#         return enc_encoding
#
#     def _reorder_cache(self, past, beam_idx):
#         reordered_past = ()
#         for layer_past in past:
#             # cached cross_attention states don't have to be reordered -> they are always the same
#             reordered_past += (
#                 tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
#             )
#         return reordered_past
#
#     def prepare_inputs_for_generation(self, dec_ids, past_key_values=None,
#                                       use_cache=None, enc_encoding=None, **kwargs):
#         # cut decoder_input_ids if past is used
#         if past_key_values is not None:
#             dec_ids = dec_ids[:, -1:]
#
#         return {
#             "enc_ids": None,  # encoder_outputs is defined. input_ids not needed
#             "enc_encoding": enc_encoding,
#             "past_key_values": past_key_values,
#             "dec_ids": dec_ids,
#             "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
#         }

