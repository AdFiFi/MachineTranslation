import os
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class Tokenizers(nn.Module):
    def __init__(self, src_language, tgt_language):
        super(Tokenizers, self).__init__()
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.language_index = {self.src_language: 0, self.tgt_language: 1}
        # Place-holders
        self.token_transform = {}
        self.vocab_transform = {}
        self.text_transform = {}

    def forward(self):
        pass

    def yield_tokens(self, data_iter: Iterable, language: str) -> List[str]:
        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[self.language_index[language]])

    def build(self):
        def sequential_transforms(*transforms):
            # helper function to club together sequential operations
            def func(txt_input):
                for transform in transforms:
                    txt_input = transform(txt_input)
                return txt_input

            return func

        self.token_transform[self.src_language] = get_tokenizer('spacy', language='de_core_news_md')
        self.token_transform[self.tgt_language] = get_tokenizer('spacy', language='en_core_web_md')

        for ln in [self.src_language, self.tgt_language]:
            # Training data Iterator
            train_iter = Multi30k(root="/data", split='train', language_pair=(self.src_language, self.tgt_language))
            # Create torchtext's Vocab object
            self.vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(train_iter, ln),
                                                                 min_freq=1,
                                                                 specials=special_symbols,
                                                                 special_first=True)

        # Set UNK_IDX as the default index. This index is returned when the token is not found.
        # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
        for ln in [self.src_language, self.tgt_language]:
            self.vocab_transform[ln].set_default_index(UNK_IDX)

        # src and tgt language text transforms to convert raw strings into tensors indices
        for ln in [self.src_language, self.tgt_language]:
            self.text_transform[ln] = sequential_transforms(self.token_transform[ln],  # Tokenization
                                                            self.vocab_transform[ln],  # Numericalization
                                                            self.tensor_transform)  # Add BOS/EOS and create tensor

    @staticmethod
    def tensor_transform(token_ids: List[int]):
        # function to add BOS/EOS and create tensor for input sequence indices
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))

    def tokenize(self, sample, language):
        return self.text_transform[language](sample)

    # def save_model(self, save_dir):
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     torch.save(self, os.path.join(save_dir, "tokenizer.bin"))

    # @classmethod
    # def load(cls, save_dir):
    #     return torch.load(os.path.join(save_dir, "tokenizer.bin"))


if __name__ == '__main__':
    tok = Tokenizers('de', 'en')
    tok.build()
    # tok.save_model('../output_dir')
