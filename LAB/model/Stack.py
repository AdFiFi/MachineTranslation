import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    LogitsProcessorList,
    StoppingCriteriaList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
    MaxLengthCriteria
)

from .Embed import DataEmbedding
from .Stack_EncDec import StackEncoder, StackDecoder


class StackConfig:
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
                 pad_token_id=0,
                 bos_token_id=2,
                 eos_token_id=3,
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


class Stack(nn.Module):
    def __init__(self, config: StackConfig):
        super().__init__()
        self.config = config
        self.output_attention = config.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(config.enc_vocab_size, config.d_model, config.dropout)
        self.dec_embedding = DataEmbedding(config.dec_vocab_size, config.d_model, config.dropout)
        # Encoder
        self.encoder = StackEncoder(config)
        # Decoder
        self.decoder = StackDecoder(config)

        # self.init_parameters()

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
                nn.init.normal_(p, mean=0, std=1)

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
        logits = None
        for i in range(max_len - 1):
            logits = self.decode(enc_encoding, dec_ids)
            next_ids = logits[:, i]
            next_ids = next_ids.data.max(-1, keepdim=True)[1]
            dec_ids = torch.cat((dec_ids, next_ids), dim=1)
            if all(torch.logical_or(next_ids == self.config.pad_token_id, next_ids == self.config.eos_token_id)):
                break

        return dec_ids, logits

    def beam_generate(self, enc_ids, enc_padding_mask, dec_ids, max_len=None, enc_attn_mask=None, num_beams=5):
        logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(5, eos_token_id=self.config.eos_token_id), ])
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=self.config.max_seq_len))
        cur_batch_size = enc_ids.size(0)
        enc_ids = enc_ids.repeat_interleave(num_beams, dim=0)
        enc_padding_mask = enc_padding_mask.repeat_interleave(num_beams, dim=0)
        dec_ids = dec_ids.repeat_interleave(num_beams, dim=0)
        beam_scorer = BeamSearchScorer(
            batch_size=cur_batch_size,
            num_beams=num_beams,
            device=enc_ids.device,
        )
        beam_scores = torch.zeros((cur_batch_size, num_beams), dtype=torch.float, device=enc_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((cur_batch_size * num_beams,))

        enc_encoding, _ = self.encode(enc_ids, enc_padding_mask, enc_attn_mask)
        max_len = self.config.max_seq_len if max_len is None else min(max_len, self.config.max_seq_len)
        logits = None
        for i in range(max_len - 1):
            logits = self.decode(enc_encoding, dec_ids)
            N, T = dec_ids.size()
            outputs = logits.view((N, T, logits.size(-1)))
            next_token_logits = outputs[:, i]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(dec_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(cur_batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                dec_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.config.pad_token_id,
                eos_token_id=self.config.eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            dec_ids = dec_ids[beam_idx]
            dec_ids = torch.cat((dec_ids, beam_next_tokens.unsqueeze(1)), dim=1)
            if not any(beam_next_tokens):
                break

        sequence_outputs = beam_scorer.finalize(
            dec_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            max_length=stopping_criteria.max_length,
        )
        dec_ids = sequence_outputs["sequences"]
        return dec_ids, logits

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

