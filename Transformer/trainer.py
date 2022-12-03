import json
import os
from timeit import default_timer as timer

from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k

from model import Transformer, TransformerConfig
from utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.model_config = TransformerConfig(enc_vocab_size=args.enc_vocab_size,
                                              dec_vocab_size=args.dec_vocab_size,
                                              d_model=args.d_model,
                                              num_heads=args.num_heads,
                                              dim_feedforward=args.dim_feedforward,
                                              batch_size=args.batch_size,
                                              num_encoder_layers=args.num_encoder_layers,
                                              num_decoder_layers=args.num_decoder_layers,
                                              activation=args.activation)
        self.tokenizer = Tokenizers(args.src_language, args.tgt_language)
        self.tokenizer.build()
        self.model = Transformer(self.model_config).to(args.device)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        self.datasets = Multi30k

    def collate_fn(self, x):
        return collate_fn(x, self.tokenizer)

    def train_epoch(self):
        train_iter = self.datasets(root="./data", split='train',
                                   language_pair=(self.args.src_language, self.args.tgt_language))
        train_dataloader = DataLoader(train_iter, batch_size=self.args.batch_size, collate_fn=self.collate_fn)
        self.model.train()
        losses = 0

        for src_ids, tgt_ids in train_dataloader:
            enc_ids = src_ids.to(self.device)
            tgt_ids = tgt_ids.to(self.device)
            dec_ids = tgt_ids[:-1, :]
            enc_padding_mask, dec_padding_mask = create_mask(enc_ids, dec_ids, self.device)
            logits = self.model(enc_ids, dec_ids,
                                enc_padding_mask, dec_padding_mask)
            self.optimizer.zero_grad()
            tgt_out = tgt_ids[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            self.optimizer.step()
            losses += loss.item()
        return losses / len(train_dataloader)

    def train(self):
        for epoch in range(1, self.args.num_epochs + 1):
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()
            val_loss = self.evaluate()
            logger.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                        f"Epoch time = {(end_time - start_time):.3f}s")

    def evaluate(self):
        train_iter = self.datasets(split='valid', language_pair=(self.args.src_language, self.args.tgt_language))
        train_dataloader = DataLoader(train_iter, batch_size=self.args.batch_size, collate_fn=self.collate_fn)
        self.model.eval()
        losses = 0

        for src_ids, tgt_ids in train_dataloader:
            enc_ids = src_ids.to(self.device)
            tgt_ids = tgt_ids.to(self.device)
            dec_ids = tgt_ids[:-1, :]
            enc_padding_mask, dec_padding_mask = create_mask(enc_ids, dec_ids, self.device)
            logits = self.model(enc_ids, dec_ids,
                                enc_padding_mask, dec_padding_mask)
            tgt_out = tgt_ids[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
        return losses / len(train_dataloader)

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if self.args.parallel else self.model
        torch.save(model_to_save, os.path.join(self.args.model_dir, 'model.bin'))

        # Save training arguments together with the trained model
        args_dict = {k: v for k, v in self.args.__dict__.items()}
        with open(os.path.join(self.args.model_dir, "config.json"), 'w') as f:
            f.write(json.dumps(args_dict))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        self.model = torch.load(os.path.join(self.args.model_dir, 'model.bin'))
        self.model.to(self.device)
        if self.args.parallel:
            self.model = torch.nn.DataParallel(self.model)
        logger.info("***** Model Loaded *****")
