import os
import sys
import yaml
import random
import numpy as np
import torch
import torch.nn as nn

from typing import Optional
import logging
from logging import Logger


### metrics
def MAE(pred, true):
    return torch.abs(pred - true)


def MSE(pred, true):
    return (pred - true) ** 2


def MAPE(pred, true):
    return torch.abs((pred - true) / true)


def SMAPE(pred, true):
    # Avoid division by zero by adding a small constant
    denominator = (torch.abs(true) + torch.abs(pred)) / 2 + 1e-8

    # Calculate the SMAPE
    smape_value = torch.mean(torch.abs(pred - true) / denominator)

    return smape_value


def masked_loss(y_pred, y_true, loss_func):
    y_true[y_true < 1e-4] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean()  # assign the sample weights of zeros to nonzero-values
    loss = loss_func(y_pred, y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_rmse_loss(y_pred, y_true):
    y_true[y_true < 1e-4] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_pred - y_true, 2)
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())


def compute_all_metrics(y_pred, y_true):
    mae = masked_loss(y_pred, y_true, MAE).item()
    rmse = masked_rmse_loss(y_pred, y_true).item()
    smape = masked_loss(y_pred, y_true, SMAPE).item()
    return mae, smape, rmse


### tools

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, logger: Optional[Logger]=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            message = f'EarlyStopping counter: {self.counter} out of {self.patience}'
            self.logger.info(message)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            message = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            self.logger.info(message)
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parsing_syntax(unknown):
    unknown_dict = {}
    key = None
    for arg in unknown:
        if arg.startswith('--'):
            key = arg.lstrip('--')
            unknown_dict[key] = None
        else:
            if key:
                unknown_dict[key] = arg
                key = None
    return unknown_dict


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
            if key == 'data' and isinstance(value, str):
                dataset_config = load_config("../Model_Config/dataset_config/{}".format(value + ".yaml"))
                self[key]= ConfigDict(dataset_config)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value


def update_config(config, unknown_args):
    for key, value in unknown_args.items():
        config_path = key.split('-')
        cur = config
        for node in config_path:
            assert node in cur.keys(), "path not exist"
            if isinstance(cur[node], ConfigDict):
                cur = cur[node]
            else:
                try:
                    cur[node] = eval(value)
                except NameError:
                    cur[node] = value
    return config


def load_graph_data(dataset_path):
    npz_path = os.path.join(dataset_path, 'graph_data.npz')
    data = np.load(npz_path)

    adj_mx = data['adj_mx']
    edge_index = data['edge_index']
    edge_attr = data['edge_attr']   # {diff_dist, dist_km, direction}
    node_attr = data['node_attr']

    return adj_mx, edge_index.T, edge_attr, node_attr


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO, to_stdout=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add console handler.
    if to_stdout:
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    # Add file handler and stdout handler
    if log_dir:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M')
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('Log directory: %s', log_dir)
    return logger


def init_network_weights(net, std=0.1):
    """
    Just for nn.Linear net.
    """
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim // 2

    res = data[..., :last_dim], data[..., last_dim:]
    return res


def exchange_df_column(df, col1, col2):
    """
    exchange df column
    :return new_df
    """
    assert (col1 in df.columns) and (col2 in df.columns)
    df[col1], df[col2] = df[col2].copy(), df[col1].copy()
    df = df.rename(columns={col1: 'temp', col2: col1})
    df = df.rename(columns={'temp': col2})
    return df