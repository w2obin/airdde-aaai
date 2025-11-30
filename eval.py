import sys
import yaml
import os

gpu_list = "0"
device_map = {gpu: i for i, gpu in enumerate(gpu_list.split(','))}
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
sys.path.append('../../AirDDE')

import argparse
import torch
import random
import numpy as np
from trainer import Exp_Basic
from data_loader import *
from utils import *

class Evaluation_Air_Pollution(Exp_Basic):
    def __init__(self, args):
        adj_mx, edge_index, edge_attr, node_attr = load_graph_data(args.data.root_path)
        args.adj_mx = adj_mx    # N x N
        args.edge_index = edge_index    # 2 x M
        args.edge_attr = edge_attr      # M x D
        args.node_attr = node_attr      # N x D

        self._logger = get_logger(None, args.model_name, 'info.log',
                                  level=args.log_level, to_stdout=args.to_stdout)
        args.logger = self._logger

        if args.data.embed:
            args.model.input_dim = int(args.model.input_dim) + int(args.model.embed_dim)

        super(Evaluation_Air_Pollution, self).__init__(args)

        self.num_nodes = adj_mx.shape[0]
        self.input_var = int(self.args.model.input_dim)
        self.input_dim = int(self.args.model.X_dim)
        self.seq_len = int(self.args.model.seq_len)
        self.horizon = int(self.args.model.horizon)
        self.output_dim = int(self.args.model.X_dim)

        self.report_filepath = self.args.report_filepath
        self.result = []
        self.result.append([self.model.setting])
        self.result.append([self.model_parameters])

    def _build_model(self):
        dataset, _ = self._get_data('val')
        self.args.data.mean_ = dataset.scaler.mean_
        self.args.data.std_ = dataset.scaler.scale_
        model = self.model_dict[self.args.model_name](
            adj_mx = self.args.adj_mx, 
            edge_index = self.args.edge_index, 
            edge_attr = self.args.edge_attr, 
            node_attr = self.args.node_attr,
            wind_mean = self.args.data.mean_,
            wind_std = self.args.data.std_,
            num_nodes=int(self.args.num_nodes),
            input_dim=int(self.args.input_dim),
            output_dim=int(self.args.output_dim),
            horizon=int(self.args.horizon),
            rnn_units=int(self.args.rnn_units),
            num_layers=int(self.args.num_rnn_layers), 
            cheb_k=2,
            ycov_dim=5, 
            mem_num=20, 
            mem_dim=64, 
            cl_decay_steps=2000, 
            use_curriculum_learning=True
        ).float()
        self.model_parameters = count_parameters(model)
        if self.args.GPU.use_multi_gpu and self.args.GPU.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.GPU.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def test(self):
        test_data, test_loader = self._get_data(flag='test')
        self.inverse_transform = test_data.inverse_transform
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.model.setting, 'checkpoint.pth')))

        with torch.no_grad():
            self.model.eval()

            truths = []
            preds = []
            batches_seen = 0
            for _, (x, gt) in enumerate(test_loader):
                x, gt, y_embed = self._prepare_data(x, gt)
                output = self.model(x, y_embed, gt,batches_seen)

                truths.append(gt.cpu().permute(1, 0, 2))   # B x T x N
                preds.append(output.cpu().permute(1, 0, 2))

            truths = torch.cat(truths, dim=0)   # B x T x N
            preds = torch.cat(preds, dim=0)

            all_mae = []
            all_smape = []
            all_rmse = []

            assert self.horizon == 24
            for i in range(0, self.horizon, 8):
                pred = preds[:, i: i + 8]
                truth = truths[:, i: i + 8]
                mae, smape, rmse = self._compute_loss_eval(truth, pred)
                all_mae.append(mae)
                all_smape.append(smape)
                all_rmse.append(rmse)
                self._logger.info('Evaluation {}h-{}h: - mae - {:.4f} - rmse - {:.4f} - smape - {:.4f}'.format(
                    i*3, (i+8)*3, mae, rmse, smape))

            # three days
            mae, smape, rmse = self._compute_loss_eval(truths, preds)
            all_mae.append(mae)
            all_smape.append(smape)
            all_rmse.append(rmse)
            self._logger.info('Evaluation all: - mae - {:.4f} - rmse - {:.4f} - smape - {:.4f}'.format(
                mae, rmse, smape))

            all_metrics = {'mae': all_mae, 'rmse': all_rmse, 'smape': all_smape}

            test_res = list(np.array([v for k, v in all_metrics.items()]).T.flatten())
            self.result.append(list(map(lambda x: round(x, 4), test_res)))

            truths_scaled = self.inverse_transform(truths).numpy()
            preds_scaled = self.inverse_transform(preds).numpy()

            return all_mae, all_smape, all_rmse, preds_scaled, truths_scaled

    def vali(self):
        vali_data, vali_loader = self._get_data(flag='val')
        self.inverse_transform = vali_data.inverse_transform
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.model.setting, 'checkpoint.pth')))

        with torch.no_grad():
            self.model.eval()

            truths = []
            preds = []
            batches_seen = 0
            for i, (x, gt) in enumerate(vali_loader):
                x, gt, y_embed = self._prepare_data(x, gt)

                output = self.model(x, y_embed, gt,batches_seen)
                batches_seen += 1
                truths.append(gt.cpu().permute(1, 0, 2))    # B x T x N
                preds.append(output.cpu().permute(1, 0, 2))

            truths = torch.cat(truths, dim=0)
            preds = torch.cat(preds, dim=0)

            mae, smape, rmse = self._compute_loss_eval(truths, preds)

            self._logger.info('Evaluation: - mae - {:.4f} - smape - {:.4f} - rmse - {:.4f}'
                              .format(mae, smape, rmse))
            val_res = [mae, rmse, smape]
            self.result.append(list(map(lambda x: round(x, 4), val_res)))

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)  # B x 24(72 hours) x N x D
        x, y, y_embed = self._get_x_y_in_correct_dims(x, y)  # 24 x B x N x D
        return x.to(self.device), y.to(self.device), y_embed.to(self.device)  # 24 x B x 35 * 11

    def _get_x_y(self, x, y):
        x = x.float()
        y = y.float()
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        # print(2333)
        # print(x.shape)
        # print(y.shape)
        batch_size = x.size(1)
        if self.args.data.embed:
            station_x = torch.arange(0, self.num_nodes).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(self.seq_len, batch_size, 1, 1)
            station_y = torch.arange(0, self.num_nodes).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(self.horizon, batch_size, 1, 1)
            x = torch.cat([x, station_x], dim=-1)
            y = torch.cat([y, station_y], dim=-1)
            x = x.reshape(self.seq_len, batch_size, self.num_nodes * self.input_var)
            embed = [6, 7, 8, 9, 10, 11]
            y_embed = y[..., embed].reshape(self.horizon, batch_size, self.num_nodes*len(embed))
            y = y[..., :self.output_dim].reshape(self.horizon, batch_size,
                                              self.num_nodes*self.output_dim)
        else:
            x0 = x[..., :self.input_var].reshape(self.seq_len, batch_size, self.num_nodes * self.input_var)

            y0 = y[..., :self.output_dim].reshape(self.horizon, batch_size,
                                                 self.num_nodes * self.output_dim)

            y_embed = y[..., self.output_dim:].reshape(self.horizon, batch_size, self.num_nodes, -1)
        return x0, y0, y_embed

    def _compute_loss_eval(self, y_true, y_predicted):
        y_true = self.inverse_transform(y_true)
        y_predicted = self.inverse_transform(y_predicted)
        return compute_all_metrics(y_predicted, y_true)


def get_mean_std(data_list):
    return data_list.mean(), data_list.std()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoAir')

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
    parser.add_argument('--report_filepath', type=str, default="/home/wbq/AIRforcast/MegaCRN/Run/")
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
        print('\nNo%d experiment ~~~' % exp_idx)

        exp = Evaluation_Air_Pollution(args)
        # exp.vali()

        # 测试评估
        mae, mape, rmse, preds, truths = exp.test()
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)

    mae_list = np.array(mae_list)  # num_exp x num_seq
    mape_list = np.array(mape_list)
    rmse_list = np.array(rmse_list)


    np.save('logs/prediction.npy',preds)
    np.save('logs/truths.npy',truths)

    seq_len = [(0, 8), (8, 16), (16, 24)]  # seq_len * 3小时（3小时一个点）
    output_text = ''
    output_text += '--------- Final Results ------------\n'
    output_text += 'MAE {}\n'.format(mae_list)
    output_text += 'MAPE {}\n'.format(mape_list)
    output_text += 'RMSE {}\n'.format(rmse_list)
    for i, (start, end) in enumerate(seq_len):
        output_text += 'Evaluation seq {}h-{}h:\n'.format(start, end)
        output_text += 'MAE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(mae_list[:, i])[0],
                                                                 get_mean_std(mae_list[:, i])[1])
        output_text += 'SMAPE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(mape_list[:, i])[0],
                                                                  get_mean_std(mape_list[:, i])[1])
        output_text += 'RMSE | mean: {:.4f} std: {:.4f}\n\n'.format(get_mean_std(rmse_list[:, i])[0],
                                                                    get_mean_std(rmse_list[:, i])[1])

    # Write the output text to a file
    with open('logs/results.txt', 'a') as file:
        file.write(output_text)
