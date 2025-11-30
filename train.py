import sys
import yaml
import os

gpu_list = "0"
device_map = {gpu: i for i, gpu in enumerate(gpu_list.split(','))}
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
sys.path.append('../AirDDE')

import argparse
import torch
import random
from utils import parsing_syntax, ConfigDict, load_config, update_config, fix_seed
from trainer import Exp_Air


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AirDDE')

    parser.add_argument('--config_filename', type=str, default='./configs/knowair_config.yaml', help='Configuration yaml file')
    parser.add_argument('--itr', type=int, default=1, help='Number of experiments.')
    parser.add_argument('--random_seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--des', type=str, default='1', help="description of experiment.")
    parser.add_argument('--num_nodes', type=int, default=184, help='num_nodes')
    parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--horizon', type=int, default=24, help='output sequence length')
    parser.add_argument('--input_dim', type=int, default=6, help='number of input channel')
    parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
    parser.add_argument('--max_diffusion_step', type=int, default=3, help='max diffusion step or Cheb K')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of rnn layers')
    parser.add_argument('--rnn_units', type=int, default=64, help='number of rnn units')
    parser.add_argument('--mem_num', type=int, default=20, help='number of meta-nodes/prototypes')
    parser.add_argument('--mem_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
    parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
    parser.add_argument('--lamb', type=float, default=0.01, help='lamb value for separate loss')
    parser.add_argument('--lamb1', type=float, default=0.01, help='lamb1 value for compact loss')
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--patience", type=int, default=20, help="patience used for early stop")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.01, help="base learning rate")
    parser.add_argument("--steps", type=eval, default=[50, 100], help="steps")
    parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
    parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
    parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
    parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
    parser.add_argument('--test_every_n_epochs', type=int, default=5, help='test_every_n_epochs')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')

    args, unknown = parser.parse_known_args()
    unknown = parsing_syntax(unknown)

    config = load_config(args.config_filename)
    config = ConfigDict(config)
    config = update_config(config, unknown)
    for attr, value in config.items():
        setattr(args, attr, value)

    # random seed
    fix_seed(args.random_seed)

    args.GPU.use_gpu = True if torch.cuda.is_available() and args.GPU.use_gpu else False

    if args.GPU.use_gpu and not args.GPU.use_multi_gpu:
        try:
            args.GPU.gpu = device_map[str(args.GPU.gpu)]
        except KeyError:
            raise KeyError("This GPU isn't available.")

    if args.GPU.use_gpu and args.GPU.use_multi_gpu:
        args.GPU.devices = args.GPU.devices.replace(' ', '')
        device_ids = args.GPU.devices.split(',')
        args.GPU.device_ids = [int(id_) for id_ in device_ids]
        args.GPU.gpu = args.GPU.device_ids[0]

    rmse_list, mae_list, mape_list = [], [], []
    for exp_idx in range(args.itr):
        args.exp_idx = exp_idx
        if args.to_stdout:
            print('\nNo%d experiment ~~~' % exp_idx)

        exp = Exp_Air(args)
        exp.train()
        torch.cuda.empty_cache()
