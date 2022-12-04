from torch.nn.utils.rnn import pad_sequence

from .tokenizer import PAD_IDX


# function to collate data samples into batch tensors
def collate_fn(batch, tokenizer, max_seq_len):
    src_batch, tgt_batch = [], []
    for sample in batch:
        # src_batch.append(tokenizer.tokenize(src_sample.rstrip("\n"), tokenizer.src_language))
        # tgt_batch.append(tokenizer.tokenize(tgt_sample.rstrip("\n"), tokenizer.tgt_language))
        src_sample = sample['translation'][tokenizer.src_language]
        tgt_sample = sample['translation'][tokenizer.tgt_language]
        src_batch.append(tokenizer.encode_expand(src_sample.rstrip("\n"), max_seq_len))
        tgt_batch.append(tokenizer.encode_expand(tgt_sample.rstrip("\n"), max_seq_len))

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch
