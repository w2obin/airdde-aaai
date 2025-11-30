import os
import numpy as np
import pandas as pd
from metpy.units import units
import metpy.calc as mpcalc
from typing import Union, List
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import chinese_calendar as calendar


class Dataset_KnowAir(Dataset):
    def __init__(self, root_path, flag='train', seq_len=24, pred_len=24,
                 freq='3h', scale=True, embed=0,
                 normalized_col: Union[str, List[int]]='default'):
        if normalized_col == 'default':
            self.normalized_col = np.arange(0, 6)
        else:
            self.normalized_col = normalized_col

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_size = seq_len + pred_len
        self.scale = scale
        self.embed = embed
        if scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.station_info = pd.read_csv(os.path.join(self.root_path, "station.csv"))
        self.stations_npy = os.path.join(self.root_path, "KnowAir.npy")
        metero_var = ['100m_u_component_of_wind', '100m_v_component_of_wind', '2m_dewpoint_temperature',
                       '2m_temperature', 'boundary_layer_height', 'k_index', 'relative_humidity+950',
                       'relative_humidity+975', 'specific_humidity+950', 'surface_pressure',
                       'temperature+925', 'temperature+950', 'total_precipitation', 'u_component_of_wind+950',
                       'v_component_of_wind+950', 'vertical_velocity+950', 'vorticity+950']
        metero_use = ['2m_temperature', 'surface_pressure', 'relative_humidity+950',
                      '100m_u_component_of_wind', '100m_v_component_of_wind']
        self.metero_idx = [metero_var.index(var) for var in metero_use]
        self.time_idx = pd.date_range(start='2015-01-01', end='2018-12-31 21:00', freq='3H')

        self.__process_raw_data__()
        self.__read_data__()

    def __process_raw_data__(self):
        raw_data = np.load(self.stations_npy)
        self.pm25 = raw_data[:, :, -1:]
        self.feature = raw_data[:, :, :-1]
        self.feature = self.feature[:, :, self.metero_idx]
        u = self.feature[:, :, -2] * units.meter / units.second   # m/s
        v = self.feature[:, :, -1] * units.meter / units.second   # m/s
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude    # km/h
        direc = mpcalc.wind_direction(u, v)._magnitude
        self.feature[:, :, -2] = speed
        self.feature[:, :, -1] = direc

        self.raw_data = np.concatenate([self.pm25, self.feature], axis=-1)  # T x N x D

    def __read_data__(self):
        # 2:1:1
        border1s = [0, int(len(self.raw_data) * 0.5), int(len(self.raw_data) * 0.75)]
        border2s = [int(len(self.raw_data) * 0.5), int(len(self.raw_data) * 0.75), len(self.raw_data)]
        self.train_border = (border1s[0], border2s[0])
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.data = self.raw_data[border1: border2]
        if self.embed:
            self.time_info = self.cal_time_info(self.time_idx[border1: border2]).values

        if self.scale:
            train_set = self.raw_data[self.train_border[0]: self.train_border[1], :, :]
            T, N, D = self.data.shape
            self.scaler.fit(train_set.reshape(-1, D)[:, self.normalized_col])
            self.data = self.data.reshape(-1, D)
            self.data[:, self.normalized_col] = self.scaler.transform(self.data[:, self.normalized_col])
            self.data = self.data.reshape(T, N, D)

    def cal_time_info(self, time_idx):
        def check_holiday(date):
            return 1 if calendar.is_holiday(date) or calendar.is_in_lieu(date) else 0

        time_info = pd.DataFrame({
            'time': time_idx,
            'hour_of_day': time_idx.hour,  # hour-day
            'day_of_week': time_idx.dayofweek,  # day-week
            'day_of_month': time_idx.day - 1,  # day-month
            'month_of_year': time_idx.month - 1,  # month-year
        })
        time_info['is_holiday'] = [check_holiday(d.date()) for d in time_idx]

        time_info.set_index('time', inplace=True)
        return time_info

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        x_start = idx
        x_end = x_start + self.seq_len
        y_start = idx + self.seq_len
        y_end = y_start + self.pred_len

        seq_x = self.data[x_start: x_end]
        seq_y = self.data[y_start: y_end]
        if self.embed:
            seq_x_time_info = self.time_info[x_start: x_end]
            seq_x_time_info = np.expand_dims(seq_x_time_info, axis=1).repeat(seq_x.shape[1],axis=1)
            seq_x = np.concatenate([seq_x, seq_x_time_info], axis=2)

            seq_y_time_info = self.time_info[y_start: y_end]
            seq_y_time_info = np.expand_dims(seq_y_time_info, axis=1).repeat(seq_x.shape[1],axis=1)
            seq_y = np.concatenate([seq_y, seq_y_time_info], axis=2)

        return seq_x, seq_y

    def inverse_transform(self, data):
        assert self.scale is True
        pm25_mean = self.scaler.mean_[0]
        pm25_std = self.scaler.scale_[0]
        return (data * pm25_std) + pm25_mean


def data_provider(args, flag):
    data_args = args.data
    model_args = args.model
    Data = data_dict[data_args.data_name]

    if flag == 'train':
        shuffle_flag = True
        drop_last = True
    else:
        shuffle_flag = False
        drop_last = False
    batch_size = data_args.batch_size

    if data_args.data_name == "Beijing1718_old":
        data_set = Data(
            root_path=data_args.root_path,
            flag=flag
        )
    else:
        data_set = Data(
            root_path=data_args.root_path,
            flag=flag,
            seq_len=model_args.seq_len,
            pred_len=model_args.horizon,
            freq=data_args.interval,
            embed=data_args.embed,
            scale=True,
            normalized_col=data_args.normalized_columns
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=data_args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

data_dict = {
    'KnowAir': Dataset_KnowAir
}
