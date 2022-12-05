import json
import os
from timeit import default_timer as timer
from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler

from model import Transformer, TransformerConfig
from utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.task = f'{args.src_language}-{args.tgt_language}'
        self.device = 'cuda' if args.device != 'cpu' and torch.cuda.is_available() else args.device
        self.tokenizer = Tokenizer(args.src_language, args.tgt_language).load(os.path.join(args.model_dir, self.task))

        self.model_config = TransformerConfig(enc_vocab_size=self.tokenizer.get_vocab_size(),
                                              dec_vocab_size=self.tokenizer.get_vocab_size(),
                                              max_seq_len=args.max_seq_len,
                                              d_model=args.d_model,
                                              num_heads=args.num_heads,
                                              dim_feedforward=args.dim_feedforward,
                                              num_encoder_layers=args.num_encoder_layers,
                                              num_decoder_layers=args.num_decoder_layers,
                                              activation=args.activation)

        self.model = Transformer(self.model_config).to(args.device)
        if args.do_parallel:
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=[d for d in range(1, torch.cuda.device_count()-1)],
                                               output_device=0)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=args.learning_rate,
                                          betas=(args.beta1, args.beta2),
                                          eps=args.epsilon)
        self.scheduler = get_vanilla_schedule_with_warmup(self.optimizer, d_model=args.d_model,
                                                          num_warmup_steps=args.warmup_steps)

        # self.datasets = Multi30k
        self.datasets = load_dataset('wmt14', self.task)

    def collate_fn(self, x):
        return collate_fn(x, self.tokenizer, self.model_config.max_seq_len)

    def empty_cache(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def train_epoch(self):
        train_datasets = self.datasets['train']
        sampler = RandomSampler(train_datasets)
        train_dataloader = DataLoader(train_datasets,
                                      sampler=sampler,
                                      batch_size=self.args.train_batch_size,
                                      collate_fn=self.collate_fn,
                                      num_workers=10)
        self.model.train()
        losses = 0
        loss_list = []

        for step, (src_ids, tgt_ids) in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            enc_ids = src_ids.to(self.device)
            tgt_ids = tgt_ids.to(self.device)
            dec_ids = tgt_ids[:, :-1]
            enc_padding_mask, dec_padding_mask = create_mask(enc_ids, dec_ids, self.device)
            logits = self.model(enc_ids, dec_ids,
                                enc_padding_mask, dec_padding_mask)
            self.optimizer.zero_grad()
            tgt_out = tgt_ids[:, 1:]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            print(f"Train loss: {loss.item():.5f}")
            if step // self.args.save_steps == 0:
                self.save_model()
        return losses / len(loss_list)

    def train(self):
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            start_time = timer()
            train_loss = self.train_epoch()
            self.empty_cache()
            end_time = timer()

            if self.args.do_evaluate:
                val_loss = self.evaluate()
                self.empty_cache()
                logger.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                            f"Epoch time = {(end_time - start_time):.3f}s")
            self.save_model()

    def evaluate(self):
        evaluate_datasets = self.datasets['train']
        evaluate_dataloader = DataLoader(evaluate_datasets,
                                         batch_size=self.args.evaluate_batch_size,
                                         collate_fn=self.collate_fn,
                                         num_workers=10)
        self.model.eval()
        losses = 0
        loss_list = []
        with torch.no_grad():
            for src_ids, tgt_ids in evaluate_dataloader:
                enc_ids = src_ids.to(self.device)
                tgt_ids = tgt_ids.to(self.device)
                dec_ids = tgt_ids[:, :-1]
                enc_padding_mask, dec_padding_mask = create_mask(enc_ids, dec_ids, self.device)
                logits = self.model(enc_ids, dec_ids,
                                    enc_padding_mask, dec_padding_mask)
                tgt_out = tgt_ids[:, 1:]
                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                losses += loss.item()
                loss_list.append(loss.item())
                print(f"Evaluate loss: {loss.item():.5f}")
        return losses / len(loss_list)

    def save_model(self):
        # Save model checkpoint (Overwrite)
        path = os.path.join(self.args.model_dir, self.task)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = self.model.module if self.args.do_parallel else self.model
        torch.save(model_to_save, os.path.join(path, 'model.bin'))

        # Save training arguments together with the trained model
        args_dict = {k: v for k, v in self.args.__dict__.items()}
        with open(os.path.join(path, "config.json"), 'w') as f:
            f.write(json.dumps(args_dict))
        logger.info("Saving model checkpoint to %s", path)

    def load_model(self):
        path = os.path.join(self.args.model_dir, self.task, 'model.bin')
        if not os.path.exists(path):
            return
            # raise Exception("Model doesn't exists! Train first!")

        self.model = torch.load(os.path.join(path, 'model.bin'))
        self.model.to(self.device)
        if self.args.do_parallel:
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=[d for d in range(1, torch.cuda.device_count()-1)],
                                               output_device=0)
        logger.info("***** Model Loaded *****")
