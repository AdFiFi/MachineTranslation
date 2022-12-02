import torch
from .tokenizers import PAD_IDX


def create_mask(src, tgt, device):
    src_padding_mask = (src == PAD_IDX).transpose(0, 1).to(device)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1).to(device)
    return src_padding_mask, tgt_padding_mask


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

