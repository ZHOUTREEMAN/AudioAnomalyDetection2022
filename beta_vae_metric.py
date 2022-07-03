# -*- coding: utf-8 -*-
# @Description：
# @Author：XingZhou
# @Time：2022/7/1 11:22
# @Email：329201962@qq.com

import matplotlib
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from dataset_loaders import WaterPipeData, WaterPipeDataMfcc
from torch.autograd import Variable
import seaborn as sns

matplotlib.use('Agg')

""" feature:{t-f,mfcc,gfcc,cnn}"""
feature = 'mfcc'
root_dir = 'data/betavae'
test_dir = 'test'
if feature == 't-f':
    input_size = 224 * 224
    test_dataset = WaterPipeData(root_dir, test_dir)
elif feature == 'mfcc':
    input_size = 44 * 86
    test_dataset = WaterPipeDataMfcc(root_dir, test_dir)
test_loader = DataLoader(test_dataset, 1)
print(test_dataset.__len__())

model = torch.load("./model/noise_{}_beta_vae_auto_encoder_epoch40000_batch148.pth".format(feature))
data = torch.Tensor(1, 28 * 28)
data = Variable(data)
loss_f = torch.nn.MSELoss()
if torch.cuda.is_available():
    net = model.cuda()
    data = data.cuda()
    loss_f = loss_f.cuda()

net.eval()
z_1 = []
z_3 = []


for step, (x, tag, name) in enumerate(test_loader, 1):
    with torch.no_grad():
        raw_x = x[0]
        x = torch.reshape(x, ((1, 1, input_size)))
        data.resize_(x.size()).copy_(x)
        results = net(data)
        if tag[0] == '1':
            z_1.append(results[4][0][0].cpu().numpy())
        else:
            z_3.append(results[4][0][0].cpu().numpy())

z_1 = np.array(z_1)
z_3 = np.array(z_3)
a=z_1.mean(axis=0)
b=z_3.mean(axis=0)
output=a-b
print(output)
np.savetxt(root_dir+'/'+'metric_{}.txt'.format(feature),output,fmt='%.09f')

