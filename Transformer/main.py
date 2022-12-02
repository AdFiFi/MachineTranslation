import argparse

from trainer import Trainer
from utils import *

logger = logging.getLogger(__name__)


def main(args):
    init_logger(f'{args.log_dir}/train.log')
    trainer = Trainer(args)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    global_group = parser.add_argument_group(title="global", description="")
    global_group.add_argument("--log_dir", default="./log_dir", type=str, help="")

    model_group = parser.add_argument_group(title="model", description="")
    model_group.add_argument("--enc_vocab_size", default=0, type=int, help="")
    model_group.add_argument("--dec_vocab_size", default=0, type=int, help="")
    model_group.add_argument("--emb_size", default=512, type=int, help="")
    model_group.add_argument("--num_heads", default=8, type=int, help="")
    model_group.add_argument("--dim_feedforward", default=512, type=int, help="")
    model_group.add_argument("--batch_size", default=128, type=int, help="")
    model_group.add_argument("--num_encoder_layers", default=3, type=int, help="")
    model_group.add_argument("--num_decoder_layers", default=3, type=int, help="")
    model_group.add_argument("--activation", default="gelu", type=str, help="")

