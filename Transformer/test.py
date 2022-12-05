from model import *

if __name__ == '__main__':
    m = Transformer(TransformerConfig())
    a = m.generate(max_length=5)
