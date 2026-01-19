import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd

from mlp import SimpleMLP
from cnn import SimpleCNN
from preprocess import preprocess
import utils

# tensorboard 记录的文件夹名称
run_name = 'cnn'

# 超参数
num_epochs = 20
lr = 0.01
batch_size = 500

input_dim = 28 * 28
hidden_dim = 16
hidden_num = 2


def main():
    # 选择设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # 读入处理后的数据
    train_data, train_label, test_data = preprocess('1-4/dataset/train.csv', '1-4/dataset/test.csv')
    n_train = train_data.shape[0]
    n_test = test_data.shape[0]
    
    X_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_label, dtype=torch.int8).reshape(-1, 1)
    X_test = torch.tensor(test_data, dtype=torch.float32)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    # 从训练集随机挑选9张图片绘制
    utils.draw_imgs(X_train, y_train)
    
    # TODO: 开始你的表演，请不要直接复制代码！
    
    # 保存到 CSV 文件，第一列为图片id，第二列为预测类别
    sub = pd.DataFrame({'ImageId': np.arange(1, n_test + 1), 'Label': y_pred.cpu().numpy()})
    print(sub)
    sub.to_csv(f'1-4/{run_name}_submission.csv', index=False)


if __name__ == '__main__':
    main()
