import argparse

from torch import distributed

from trainer import Trainer
from utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
logger = logging.getLogger(__name__)


def main(args):
    local_rank = 0
    world_size = 0
    if args.do_parallel:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        distributed.init_process_group('nccl', world_size=world_size, rank=rank)
        # distributed.init_process_group('gloo', world_size=self.world_size, rank=self.rank)
        torch.cuda.set_device(local_rank)
    trainer = Trainer(args, local_rank=local_rank, world_size=world_size)
    if args.do_train:
        init_logger(f'{args.log_dir}/train_{args.model}_{args.task}.log')
        trainer.train()
    else:
        if args.do_evaluate:
            init_logger(f'{args.log_dir}/{args.model}_evaluate.log')
            trainer.load_model()
            trainer.evaluate()
        if args.do_test:
            init_logger(f'{args.log_dir}/{args.model}_test.log')
            trainer.load_model()
            trainer.test()

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    global_group = parser.add_argument_group(title="global", description="")
    global_group.add_argument("--log_dir", default="./log_dir", type=str, help="")
    global_group.add_argument("--task", default="wmt14_de_en", type=str, help="")
    global_group.add_argument("--model", default="Transformer", type=str, help="")

    data_group = parser.add_argument_group(title="data", description="")
    data_group.add_argument("--datasets", default='wmt14', type=str, help="")
    data_group.add_argument("--data_dir", default="./data", type=str, help="")
    data_group.add_argument("--src_language", default="de", type=str, help="")
    data_group.add_argument("--tgt_language", default="en", type=str, help="")
    data_group.add_argument("--data_processors", default=0, type=int, help="")

    model_group = parser.add_argument_group(title="model", description="")
    model_group.add_argument("--d_model", default=512, type=int, help="")
    model_group.add_argument("--num_heads", default=8, type=int, help="")
    model_group.add_argument("--dim_feedforward", default=2048, type=int, help="")
    model_group.add_argument("--num_encoder_layers", default=6, type=int, help="")
    model_group.add_argument("--num_decoder_layers", default=6, type=int, help="")
    model_group.add_argument("--max_seq_len", default=128, type=int, help="")
    model_group.add_argument("--activation", default="gelu", type=str, help="")
    model_group.add_argument("--model_dir", default="output_dir", type=str, help="")

    train_group = parser.add_argument_group(title="train", description="")
    train_group.add_argument("--do_train", action="store_true", help="")
    train_group.add_argument("--do_parallel", action="store_true", help="")
    train_group.add_argument("--device", default="cuda", type=str, help="")
    train_group.add_argument("--train_batch_size", default=128, type=int, help="")
    train_group.add_argument("--num_epochs", default=4, type=int, help="")
    train_group.add_argument("--learning_rate", default=1e-5, type=float, help="")
    train_group.add_argument("--beta1", default=0.9, type=float, help="")
    train_group.add_argument("--beta2", default=0.98, type=float, help="")
    train_group.add_argument("--epsilon", default=1e-9, type=float, help="")
    train_group.add_argument("--schedule", default='linear', type=str, help="")
    train_group.add_argument("--warmup_steps", default=4000, type=int, help="")
    train_group.add_argument("--save_steps", default=4000, type=int, help="")
    train_group.add_argument("--test_steps", default=1000, type=int, help="")
    train_group.add_argument("--epsilon_ls", default=0.1, type=float, help="label smoothing")

    evaluate_group = parser.add_argument_group(title="evaluate", description="")
    evaluate_group.add_argument("--do_evaluate", action="store_true", help="")
    evaluate_group.add_argument("--do_test", action="store_true", help="")
    evaluate_group.add_argument("--evaluate_batch_size", default=64, type=int, help="")
    evaluate_group.add_argument("--num_beams", default=5, type=int, help="")
    evaluate_group.add_argument("--alpha", default=0.6, type=float, help="length penalty")

    Args = parser.parse_args()
    main(Args)
