import argparse

from trainer import Trainer
from utils import *

logger = logging.getLogger(__name__)


def main(args):
    init_logger(f'{args.log_dir}/train.log')
    trainer = Trainer(args)
    trainer.train()

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    global_group = parser.add_argument_group(title="global", description="")
    global_group.add_argument("--log_dir", default="./log_dir", type=str, help="")

    data_group = parser.add_argument_group(title="data", description="")
    data_group.add_argument("--data_dir", default="./data", help="")
    data_group.add_argument("--src_language", default="de", help="")
    data_group.add_argument("--tgt_language", default="en", help="")

    model_group = parser.add_argument_group(title="model", description="")
    model_group.add_argument("--d_model", default=512, type=int, help="")
    model_group.add_argument("--num_heads", default=8, type=int, help="")
    model_group.add_argument("--dim_feedforward", default=512, type=int, help="")
    model_group.add_argument("--num_encoder_layers", default=3, type=int, help="")
    model_group.add_argument("--num_decoder_layers", default=3, type=int, help="")
    model_group.add_argument("--activation", default="gelu", type=str, help="")
    model_group.add_argument("--model_dir", default="result", type=str, help="")

    train_group = parser.add_argument_group(title="train", description="")
    train_group.add_argument("--device", default="cuda", type=str, help="")
    train_group.add_argument("--batch_size", default=128, type=int, help="")
    train_group.add_argument("--num_epochs", default=10, type=int, help="")
    train_group.add_argument("--beta1", default=0.9, type=float, help="")
    train_group.add_argument("--beta2", default=0.98, type=float, help="")
    train_group.add_argument("--epsilon", default=1e-9, type=float, help="")
    train_group.add_argument("--warmup_steps", default=4000, type=float, help="")

    Args = parser.parse_args()
    main(Args)
