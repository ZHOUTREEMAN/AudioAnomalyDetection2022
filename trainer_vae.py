# -*- coding: utf-8 -*-
# @Description：用于vae的训练类
# @Author：XingZhou
# @Time：2022/4/22 15:44
# @Email：329201962@qq.com
import torch
from numpy import shape
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from dataset_loaders import WaterPipeData
from torchsummary import summary
import matplotlib

from variational_auto_encoder import Autoencoder

matplotlib.use('AGG')
import matplotlib.pyplot as plt

root_dir = 'data/noise_after'
train_dir = 'train'
np.random.seed(123)
torch.manual_seed(123)
BATCH_SIZE = 64
LR = 0.000001
EPOCHS = 3000

latent_length = 30
input_size = 224 * 224
hidden1 = 512
hidden2 = 256
hidden3 = 128
hidden4 = 64

train_dataset = WaterPipeData(root_dir, train_dir)
train_loader = DataLoader(train_dataset, BATCH_SIZE, True, drop_last=True)

dataiter = iter(train_loader)
inputs, labels = dataiter.next()
net = Autoencoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)
data = torch.Tensor(BATCH_SIZE, 28 * 28)
data = Variable(data)
if torch.cuda.is_available():
    net = net.cuda()
    data = data.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_f = nn.MSELoss()
if torch.cuda.is_available():
    loss_f = loss_f.cuda()

summary(net, input_size=(1, 224 * 224), batch_size=64, device="cuda")
Loss_list_epoch = []

for epoch in range(EPOCHS):
    Loss_list = []
    net.train()
    for step, (x, _) in enumerate(train_loader, 1):
        x = torch.reshape(x, ((64, 1, 224 * 224)))
        net.zero_grad()
        data.resize_(x.size()).copy_(x)
        decoded , latent, latent_mean, latent_logvar= net(data)
        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        loss = loss_f(decoded, data)+kl_loss
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
plt.savefig("./log/loss_all_vae.jpg")
print("task over,saving model......")
torch.save(net, "./model/noise_vae_auto_encoder_epoch{}_batch{}.pth".format(EPOCHS, BATCH_SIZE))



