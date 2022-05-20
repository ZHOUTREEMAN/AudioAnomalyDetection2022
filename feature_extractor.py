# -*- coding: utf-8 -*-
# @Time : 2021/12/8 19:11
# @Author : XingZhou
# @Email : 329201962@qq.com
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_loaders import WaterPipeDataForFE


def feature_extractor(model_type, input):  # 特征提取器，返回训练好的神经网络的中间层参数
    input = torch.tensor(input)
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = torch.load("./model/noise_FE_epoch10_batch2.pth").to(device)
    model2 = torch.load("./model/noise_FE_epoch10_batch2.pth").to(device)
    model3 = torch.load("./model/noise_FE_epoch10_batch2.pth").to(device)
    input = input.to(device)
    model = {}
    model['v1'] = model1
    model['v2'] = model2
    model['v3'] = model3
    net = model[model_type]
    _, output = net(input)
    return output


if __name__ == "__main__":
    root_dir = 'data/FE'
    test_dir = 'test'
    np.random.seed(123)
    torch.manual_seed(123)
    BATCH_SIZE = 2
    LR = 0.0005
    EPOCHS = 100
    test_dataset = WaterPipeDataForFE(root_dir, test_dir)
    test_loader = DataLoader(test_dataset, 1)
    input = torch.Tensor(1, 3, 224, 224)
    out = feature_extractor('v3', input)
    print(np.shape(out))  # torch.Size([1, 960, 7, 7])
