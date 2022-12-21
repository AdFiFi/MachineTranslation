from datasets import load_dataset
from torch.utils.data import Dataset
from torchtext.datasets import multi30k, Multi30k
multi30k.MD5["test"] = "0681be16a532912288a91ddd573594fbdd57c0fbb81486eff7c55247e35326c2"


class Multi30kDatasets(Dataset):
    def __init__(self, split='train', src_language='', tgt_language=''):
        data_iter = iter(Multi30k(split=split,
                                  language_pair=(src_language, tgt_language)))
        self.data = [{"translation": {src_language: sample[0], tgt_language:sample[1]}} for sample in data_iter]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_Multi30k_datadict(src_language, tgt_language):
    return {'train': Multi30kDatasets('train', src_language, tgt_language),
            'test': Multi30kDatasets('test', src_language, tgt_language),
            'validation': Multi30kDatasets('valid', src_language, tgt_language)}


if __name__ == '__main__':
    wmt14_fr_en = load_dataset('wmt14', 'de-en')
    # wmt14_fr_en = load_dataset('wmt14', 'fr-en')

    # multi = get_Multi30k_datadict('de', 'en')
