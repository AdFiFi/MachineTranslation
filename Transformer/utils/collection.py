from torch.nn.utils.rnn import pad_sequence

from .tokenizers import PAD_IDX, Tokenizers


# function to collate data samples into batch tesors
def collate_fn(batch, tokenizer: Tokenizers):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(tokenizer.tokenize(src_sample.rstrip("\n"), tokenizer.src_language))
        tgt_batch.append(tokenizer.tokenize(tgt_sample.rstrip("\n"), tokenizer.tgt_language))

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch
