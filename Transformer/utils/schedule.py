from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup


def get_vanilla_schedule_with_warmup(optimizer: Optimizer, d_model, num_warmup_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        current_step += 1
        return d_model**(-0.5) * min(current_step**(-0.5), current_step*num_warmup_steps**(-1.5))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_schedule(args, optimizer, t_total):
    if args.schedule == 'vanilla':
        schedule = get_vanilla_schedule_with_warmup(optimizer, d_model=args.d_model,
                                                    num_warmup_steps=args.warmup_steps)
    elif args.schedule == 'linear':
        schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                   num_training_steps=t_total)
    else:
        schedule = None
    return schedule
