# -*- coding: utf-8 -*-
# @Description：使用beta vae进行采样生成新数据
# @Author：XingZhou
# @Time：2022/7/1 15:31
# @Email：329201962@qq.com
import os

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
test_dir = 'sample'
threshold = np.load('./model/threshold_{}_vae.npy'.format(feature))
if feature == 't-f':
    input_size = 224 * 224
elif feature == 'mfcc':
    input_size = 44 * 86


model = torch.load("./model/noise_{}_beta_vae_auto_encoder_epoch40000_batch148.pth".format(feature))
test_model = torch.load("./model/noise_{}_vae_auto_encoder_epoch40000_batch967.pth".format(feature))
loss_f = torch.nn.MSELoss()
if torch.cuda.is_available():
    net = model.cuda()
    net2 = test_model.cuda()
    loss_f = loss_f.cuda()

net.eval()
sample_list = net.sample(1000,0)
outputs = []

for step, (data) in enumerate(sample_list, 1):
    with torch.no_grad():
        decoded, latent, latent_mean, latent_logvar= net2(data)
        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        loss = loss_f(decoded, data) + kl_loss
        loss = loss.cpu().detach().numpy()
        output = ""
        if loss > threshold:
            output = "3"
        else:
            output = "1"
        outputs.append(output)
        print("output:{}---loss:{}---threshold:{}".format(output, loss, threshold))   
sample_list.cpu().detach().numpy().tofile(root_dir+'/'+test_dir+'/'+'sample_{}.dat'.format(feature))
outputs = np.array(outputs)
outputs.tofile(root_dir+'/'+test_dir+'/'+'label_{}.dat'.format(feature))
