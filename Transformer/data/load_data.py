from datasets import load_dataset

if __name__ == '__main__':
    wmt14_de_en = load_dataset('wmt14', 'de-en')
    # wmt14_fr_en = load_dataset('wmt14', 'fr-en')
