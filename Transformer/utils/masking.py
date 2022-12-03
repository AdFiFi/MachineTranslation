import torch
from .tokenizers import PAD_IDX


def create_mask(src, tgt, device):
    src_padding_mask = (src == PAD_IDX).to(device)
    tgt_padding_mask = (tgt == PAD_IDX).to(device)
    return src_padding_mask, tgt_padding_mask


def triangular_causal_mask(B, L, device="cpu"):
    mask_shape = [B, 1, L, L]
    with torch.no_grad():
        mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    return mask


def mask_expand(mask: torch.Tensor, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)

    return expanded_mask
