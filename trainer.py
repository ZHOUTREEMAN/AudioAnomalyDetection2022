# -*- coding: utf-8 -*-
# @Time : 2021/12/8 19:08
# @Author : XingZhou
# @Email : 329201962@qq.com
import torch
from numpy import shape
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from dataset_loaders import WaterPipeData, WaterPipeDataMfcc
from deep_auto_encoder import AutoEncoder
from torchsummary import summary
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

""" feature:{t-f,mfcc,gfcc,cnn}"""
feature = 't-f'
root_dir = 'data/noise_after'
train_dir = 'train'
np.random.seed(123)
torch.manual_seed(123)
BATCH_SIZE = 64
LR = 0.0001
EPOCHS = 3000

if feature == 't-f':
    input_size1 = 224
    input_size2 = 224
    train_dataset = WaterPipeData(root_dir, train_dir)
elif feature == 'mfcc':
    input_size1 = 44
    input_size2 = 86
    train_dataset = WaterPipeDataMfcc(root_dir, train_dir)

train_loader = DataLoader(train_dataset, BATCH_SIZE, True, drop_last=True)
dataiter = iter(train_loader)
inputs, labels, _ = dataiter.next()
net = AutoEncoder()
data = torch.Tensor(BATCH_SIZE, 28 * 28)
data = Variable(data)
if torch.cuda.is_available():
    net = net.cuda()
    data = data.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_f = nn.MSELoss()
# loss_f = nn.BCELoss()
if torch.cuda.is_available():
    loss_f = loss_f.cuda()

summary(net, input_size=(1, input_size1, input_size2), batch_size=64, device="cuda")
Loss_list_epoch = []

for epoch in range(EPOCHS):
    Loss_list = []
    net.train()
    for step, (x, _, _) in enumerate(train_loader, 1):
        x = torch.reshape(x, ((64, 1, input_size1, input_size2)))
        net.zero_grad()
        data.resize_(x.size()).copy_(x)
        code, decoded = net(data)
        loss = loss_f(decoded, data)
        print("----------------epoch:{} step:{} loss:{}----------------".format(epoch, step, loss))
        loss.backward()
        optimizer.step()
        Loss_list.append(loss.cpu().detach().numpy())

        if step % 10 == 0:
            net.eval()
            eps = Variable(inputs)
            eps = torch.reshape(eps, ((64, 1, input_size1, input_size2)))
            eps = eps.type(torch.FloatTensor)
            if torch.cuda.is_available():
                eps = eps.cuda()
            tags, fake = net(eps)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(data), len(train_loader.dataset),
                       100. * step / len(train_loader),
                loss.item()))
    Loss_list_epoch.append(Loss_list[-1])

x = range(1, len(Loss_list_epoch) + 1)
y = Loss_list_epoch
plt.plot(x, y, '.-')
plt_title = 'EPOCHS = {}; BATCH_SIZE = {}; LEARNING_RATE:{}'.format(EPOCHS, BATCH_SIZE, LR)
plt.title(plt_title)
plt.xlabel('per epoch')
plt.ylabel('loss')
plt.savefig("./log/loss_{}_deep_all.jpg".format(feature))
print("task over,saving model......")
torch.save(net, "./model/noise_{}_deep_auto_encoder_epoch{}_batch{}.pth".format(feature,EPOCHS, BATCH_SIZE))
