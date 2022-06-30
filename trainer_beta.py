# -*- coding: utf-8 -*-
# @Description：
# @Author：XingZhou
# @Time：2022/6/30 9:30
# @Email：329201962@qq.com

import torch
from numpy import shape
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from beta_vae_linear import BetaVAE
from dataset_loaders import WaterPipeData, WaterPipeDataMfcc
from torchsummary import summary
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

""" feature:{t-f,mfcc,gfcc,cnn}"""
feature = 'mfcc'
root_dir = 'data/noise_after'
train_dir = 'train'
np.random.seed(123)
torch.manual_seed(123)
BATCH_SIZE = 967
LR = 0.000001
EPOCHS = 40000

latent_length = 30
if feature == 't-f':
    input_size = 224 * 224
    train_dataset = WaterPipeData(root_dir, train_dir)
elif feature == 'mfcc':
    input_size = 44 * 86
    train_dataset = WaterPipeDataMfcc(root_dir, train_dir)

train_loader = DataLoader(train_dataset, BATCH_SIZE, True, drop_last=True)
print(train_dataset.__len__())

dataiter = iter(train_loader)
inputs, labels, _ = dataiter.next()
print(np.shape(inputs))
net = BetaVAE(input_size, latent_length, loss_type='H')
data = torch.Tensor(BATCH_SIZE, 28 * 28)
data = Variable(data)
if torch.cuda.is_available():
    net = net.cuda()
    data = data.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=LR)

summary(net, input_size=(1, input_size), batch_size=64, device="cuda")
Loss_list_epoch = []

for epoch in range(EPOCHS):
    Loss_list = []
    net.train()
    for step, (x, _, _) in enumerate(train_loader, 1):
        x = torch.reshape(x, ((BATCH_SIZE, 1, input_size)))
        net.zero_grad()
        data.resize_(x.size()).copy_(x)
        results = net(data)
        loss = net.loss_function(*results, M_N=1.0)['loss']
        print("----------------epoch:{} step:{} loss:{}----------------".format(epoch, step, loss))
        loss.backward()
        optimizer.step()
        Loss_list.append(loss.cpu().detach().numpy())
    Loss_list_epoch.append(Loss_list[-1])

x = range(1, len(Loss_list_epoch) + 1)
y = Loss_list_epoch
plt.plot(x, y, '.-')
plt_title = 'EPOCHS = {}; BATCH_SIZE = {}; LEARNING_RATE:{}'.format(EPOCHS, BATCH_SIZE, LR)
plt.title(plt_title)
plt.xlabel('per epoch')
plt.ylabel('loss')
plt.savefig("./log/loss_all_{}_beta_vae.jpg".format(feature))
print("task over,saving model......")
torch.save(net, "./model/noise_{}_beta_vae_auto_encoder_epoch{}_batch{}.pth".format(feature, EPOCHS, BATCH_SIZE))
