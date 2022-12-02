from model import Transformer, TransformerConfig


class Trainer(object):
    def __init__(self, args):
        self.model_config = TransformerConfig()
        self.model = Transformer(self.model_config)

    def load_datasets(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

