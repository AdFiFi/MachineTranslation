import os
from typing import Iterable, List

import torch
from datasets import load_dataset, Dataset, table
from tokenizers import CharBPETokenizer
from tokenizers.models import BPE
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']


class Tokenizer(CharBPETokenizer):
    def __init__(self, src_language, tgt_language, vocab=None, merges=None, **kwargs):
        super(Tokenizer, self).__init__(vocab, merges, **kwargs)
        self.src_language = src_language
        self.tgt_language = tgt_language

    def encode_expand(self, token_ids, max_seq_len):
        token_ids = self.encode(token_ids).ids
        token_ids = torch.cat((torch.tensor([BOS_IDX]),
                               torch.tensor(token_ids[:max_seq_len-2]),
                               torch.tensor([EOS_IDX])))
        return token_ids

    def train_from_datasets(self, dataset, vocab_size, save_path):
        self.train_from_iterator(iterator=iter(dataset), vocab_size=vocab_size, min_frequency=2,
                                 special_tokens=special_tokens)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_model(save_path)

    def from_file(self, vocab_filename: str, merges_filename: str, **kwargs):
        vocab, merges = BPE.read_file(vocab_filename, merges_filename)
        return Tokenizer(self.src_language, self.tgt_language, vocab, merges, **kwargs)

    def load(self, path):
        vocab_path = os.path.join(path, 'vocab.json')
        merges_path = os.path.join(path, 'merges.txt')
        tokenizer = self.from_file(vocab_path, merges_path)
        tokenizer.add_special_tokens(special_tokens=special_tokens)
        return tokenizer


def train_de_en_tokenizer():
    tok = Tokenizer('de', 'en')
    data = load_dataset('wmt14', 'de-en')['train']
    de_list = []
    en_list = []
    for pair in data:
        de_list.append(pair['translation']['de'])
        en_list.append(pair['translation']['en'])
    new_datasets = de_list+en_list
    tok.train_from_datasets(new_datasets, vocab_size=37000, save_path='../output_dir/de-en')
    return tok


def train_fr_en_tokenizer():
    tok = Tokenizer('fr', 'en')
    data = load_dataset('wmt14', 'fr-en')['train']
    de_list = []
    en_list = []
    for pair in data:
        de_list.append(pair['translation']['fr'])
        en_list.append(pair['translation']['en'])
    new_datasets = de_list+en_list

    tok.train_from_datasets(new_datasets, vocab_size=37000, save_path='../output_dir/fr-en')
    return tok





if __name__ == '__main__':
    # tok = Tokenizers('de', 'en')
    # tok.build()
    # tok.save_model('../output_dir')

    # train_fr_en_tokenizer()
    train_fr_en_tokenizer()

