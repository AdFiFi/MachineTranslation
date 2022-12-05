from model import *

if __name__ == '__main__':
    m = TransformerForTranslation(TransformerConfig())
    from transformers import BartConfig, BartModel
    a = m.generate(max_length=5)
