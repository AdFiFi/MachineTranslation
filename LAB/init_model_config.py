from model import *
from utils import PAD_IDX, BOS_IDX,EOS_IDX


def model_and_config(args, tok):
    if args.model == "Transformer":
        model_config = TransformerConfig(enc_vocab_size=tok.get_vocab_size(),
                                         dec_vocab_size=tok.get_vocab_size(),
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

        model = Transformer(model_config)
    elif args.model == "Stack":
        model_config = StackConfig(enc_vocab_size=tok.get_vocab_size(),
                                   dec_vocab_size=tok.get_vocab_size(),
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

        model = Stack(model_config)
    elif args.model == "Cube":
        model_config = CubeConfig(enc_vocab_size=tok.get_vocab_size(),
                                  dec_vocab_size=tok.get_vocab_size(),
                                  max_seq_len=args.max_seq_len,
                                  d_model=args.d_model,
                                  num_t_heads=args.num_heads,
                                  num_s_heads=1,
                                  dim_feedforward=args.dim_feedforward,
                                  num_encoder_layers=args.num_encoder_layers,
                                  num_decoder_layers=args.num_decoder_layers,
                                  activation=args.activation,
                                  pad_token_id=PAD_IDX,
                                  bos_token_id=BOS_IDX,
                                  eos_token_id=EOS_IDX)

        model = Cube(model_config)
    else:
        model = None
        model_config = None
    return model, model_config
