import json
from timeit import default_timer as timer

from torch.utils.data import DataLoader, RandomSampler
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

from data import *
from model import Transformer, TransformerConfig
from utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.task = f'{args.src_language}-{args.tgt_language}'
        self.device = 'cuda' if args.device != 'cpu' and torch.cuda.is_available() else args.device
        self.tokenizer = SharedTokenizer(args.src_language, args.tgt_language).load(os.path.join(args.model_dir, self.task))

        self.model_config = TransformerConfig(enc_vocab_size=self.tokenizer.get_vocab_size(),
                                              dec_vocab_size=self.tokenizer.get_vocab_size(),
                                              max_seq_len=args.max_seq_len,
                                              d_model=args.d_model,
                                              num_heads=args.num_heads,
                                              dim_feedforward=args.dim_feedforward,
                                              num_encoder_layers=args.num_encoder_layers,
                                              num_decoder_layers=args.num_decoder_layers,
                                              activation=args.activation,
                                              pad_token_id=PAD_IDX,
                                              bos_token_id=BOS_IDX,
                                              eos_token_id=EOS_IDX)

        self.model = Transformer(self.model_config).to(args.device)
        if args.do_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None

        self.datasets = self.load_datasets()

    def load_datasets(self):
        if self.args.datasets == 'wmt14':
            return load_dataset('wmt14', self.task)
        elif self.args.datasets == 'Multi30k':
            return get_Multi30k_datadict(self.args.src_language, self.args.tgt_language)

    def collate_fn(self, x):
        return collate_fn_with_shared_tokenizer(x, self.tokenizer, self.model_config.max_seq_len)

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
                                      num_workers=self.args.data_processors)
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
            if step % self.args.save_steps == 0:
                self.save_model()
            if (step + 1) % self.args.test_steps == 0:
                self.test(mode='single')

        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs*len(self.datasets['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.datasets['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          betas=(self.args.beta1, self.args.beta2),
                                          eps=self.args.epsilon)
        if self.args.schedule == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.args.warmup_steps,
                                                             num_training_steps=total)
        else:
            self.scheduler = get_vanilla_schedule_with_warmup(self.optimizer, d_model=self.args.d_model,
                                                              num_warmup_steps=self.args.warmup_steps)
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

        evaluate_datasets = self.datasets['validation']
        evaluate_dataloader = DataLoader(evaluate_datasets,
                                         batch_size=self.args.evaluate_batch_size,
                                         collate_fn=self.collate_fn,
                                         num_workers=self.args.data_processors)
        logger.info("***** Running evaluation on validation dataset *****")
        logger.info("  Num examples = %d", len(evaluate_datasets))
        logger.info("  Batch size = %d", self.args.evaluate_batch_size)
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

    def test(self, mode='full'):
        test_datasets = self.datasets['test']
        sampler = RandomSampler(test_datasets)
        test_dataloader = DataLoader(test_datasets,
                                     sampler=sampler,
                                     batch_size=self.args.evaluate_batch_size,
                                     collate_fn=self.collate_fn,
                                     num_workers=self.args.data_processors)
        logger.info(f"***** Running prediction on {mode} test dataset *****")
        logger.info("  Num examples = %d", len(test_datasets))
        logger.info("  Batch size = %d", self.args.evaluate_batch_size)
        logger.info("  Search strategy = %s",
                    "greedy search" if self.args.beam_num == 1 else f"beam search(beam num={self.args.beam_num})")
        self.model.eval()
        losses = 0
        loss_list = []
        text_target_list = []
        text_generation_list = []
        text_target_corpus_list = []
        text_generation_corpus_list = []

        with torch.no_grad():
            for src_ids, tgt_ids in tqdm(test_dataloader, desc="Predicting:", ncols=0):
                enc_ids = src_ids.to(self.device)
                tgt_ids = tgt_ids.to(self.device)
                dec_ids = tgt_ids[:, :1]
                tgt_out = tgt_ids[:, 1:]
                enc_padding_mask, dec_padding_mask = create_mask(enc_ids, dec_ids, self.device)

                if self.args.do_parallel:
                    dec_ids, logits = self.model.module.greedy_generate(enc_ids, enc_padding_mask, dec_ids)
                else:
                    dec_ids, logits = self.model.greedy_generate(enc_ids, enc_padding_mask, dec_ids)
                # loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                # losses += loss.item()
                # loss_list.append(loss.item())

                preds_ids = dec_ids.detach().cpu().numpy()
                target_ids = tgt_out.detach().cpu().numpy()
                for i in range(preds_ids.shape[0]):
                    text_target = self.tokenizer.decode(target_ids[i], skip_special_tokens=True)
                    text_generation = self.tokenizer.decode(preds_ids[i], skip_special_tokens=True)

                    text_target_list.append(text_target)
                    text_generation_list.append(text_generation)
                    text_target_corpus_list.append([text_target.split()])
                    text_generation_corpus_list.append(text_generation.split())
                if mode == 'single':
                    break

        blue = bleu_score(text_generation_corpus_list, text_target_corpus_list)
        results = {
            # "loss": losses / len(loss_list),
            "BLUE": blue
        }
        logger.info(f"{results}")
        text = '\n\n'.join([t + '\n' + g for t, g in zip(text_target_list, text_generation_list)])
        with open('./test.txt', mode='w', encoding='utf-8') as f:
            f.write(text)
        return results

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
            logger.info("Model doesn't exists! Train first!")
            return
            # raise Exception("Model doesn't exists! Train first!")

        self.model = torch.load(os.path.join(path))
        self.model.to(self.device)
        if self.args.do_parallel:
            self.model = torch.nn.DataParallel(self.model)
        logger.info("***** Model Loaded *****")
