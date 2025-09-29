"""
将故障数据集集成到Time-LLM框架
文件名: data_provider/fault_data_loader.py
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import json
from pathlib import Path
import torch

warnings.filterwarnings('ignore')

class Dataset_Fault_Classification(Dataset):
    """
    故障分类的数据集
    """
    
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='fault_data',
                 target='fault_label', scale=True, timeenc=0, freq='h',
                 percent=100, seasonal_patterns=None, downsample_step=1):
        
        if size == None:
            self.seq_len = 96
            self.label_len = 0
            self.pred_len = 0
        else:
            self.seq_len = size[0]
            self.label_len = size[1] 
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.downsample_step = downsample_step  # 新增：下采样步长
        
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1] if len(self.data_x.shape) > 1 else 1

    def _apply_downsampling(self, data):
        """
        对时间序列应用下采样
        
        Args:
            data: 输入数据，形状为 (seq_len, features) 或 (batch_size, seq_len, features)
        
        Returns:
            下采样后的数据
        """
        if self.downsample_step <= 1:
            return data
        
        if len(data.shape) == 2:
            # 单个序列: (seq_len, features)
            return data[::self.downsample_step]
        elif len(data.shape) == 3:
            # 批量序列: (batch_size, seq_len, features)
            return data[:, ::self.downsample_step, :]
        else:
            return data

    def __read_data__(self):
        """读取故障分类数据"""
        self.scaler = StandardScaler()
        
        data_folder = os.path.join(self.root_path, self.data_path)
        

        all_data = np.load(os.path.join(data_folder, 'fault_data.npy'), allow_pickle=True)
        all_labels = np.load(os.path.join(data_folder, 'fault_labels.npy'), allow_pickle=True)

        # 应用下采样
        if self.downsample_step > 1:
            # print(f"Applying downsampling with step {self.downsample_step}")
            if isinstance(all_data, np.ndarray) and len(all_data.shape) == 3:
                # 如果是固定形状的数组 (n_samples, seq_len, features)
                all_data = self._apply_downsampling(all_data)
            else:
                # 如果是变长序列的列表
                downsampled_data = []
                for i, data in enumerate(all_data):
                    if isinstance(data, np.ndarray):
                        downsampled = self._apply_downsampling(data)
                        downsampled_data.append(downsampled)
                    else:
                        downsampled_data.append(data)
                all_data = np.array(downsampled_data, dtype=object)

        
        # 处理变长序列
        if isinstance(all_data, np.ndarray):
            # 固定长度或截断
            processed_data = []
            processed_labels = []
            
            for data, label in zip(all_data, all_labels):
                if len(data) >= self.seq_len:
                    # 截断到指定长度
                    processed_data.append(data[:self.seq_len])
                    processed_labels.append(label)
                elif len(data) >= self.seq_len // 2:
                    # 如果长度不足但大于一半，进行填充
                    padded_data = np.pad(data, ((0, self.seq_len - len(data)), (0, 0)), 
                                       mode='constant', constant_values=0)
                    processed_data.append(padded_data)
                    processed_labels.append(label)
            
            all_data = np.array(processed_data)
            all_labels = np.array(processed_labels)
        
        # 数据集划分
        total_len = len(all_data)
        num_train = int(total_len * 0.7)
        num_val = int(total_len * 0.1)
        
        if self.set_type == 0:  # train
            self.data_x = all_data[:num_train]
            self.labels = all_labels[:num_train]
        elif self.set_type == 1:  # val
            self.data_x = all_data[num_train:num_train + num_val]
            self.labels = all_labels[num_train:num_train + num_val]
        else:  # test
            self.data_x = all_data[num_train + num_val:]
            self.labels = all_labels[num_train + num_val:]
        
        # 标准化
        if self.scale:
            # 将所有样本的所有时间步合并进行拟合
            train_data = all_data[:num_train].reshape(-1, all_data.shape[-1])
            self.scaler.fit(train_data)
            
            # 对当前集合进行转换
            original_shape = self.data_x.shape
            self.data_x = self.scaler.transform(self.data_x.reshape(-1, original_shape[-1]))
            self.data_x = self.data_x.reshape(original_shape)


    def __getitem__(self, index):
        """获取分类样本"""
        seq_x = self.data_x[index]  # (seq_len, features)
        label = self.labels[index]  # 故障标签
        
        # 生成虚拟的时间戳
        seq_x_mark = np.zeros((len(seq_x), 4))  # month, day, weekday, hour
        
        # 对于分类任务，seq_y 可以是标签
        seq_y = np.array([label])
        seq_y_mark = np.zeros((1, 4))
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)