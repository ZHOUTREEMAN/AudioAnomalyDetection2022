# -*- coding: utf-8 -*-
# @Time : 2021/12/9 14:13
# @Author : XingZhou
# @Email : 329201962@qq.com

import torch
from torch import nn
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
HIDDEN_SIZE = 30


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.connv1 = nn.Conv2d(1, 16, 4, 2, 1)
        self.connv2 = nn.Conv2d(16, 32, 4, 2, 1)
        self.connv3 = nn.Conv2d(32, 16, 3, 1, 1)

        self.en_fc = nn.Linear(16 * 10 * 107, HIDDEN_SIZE)
        self.de_fc = nn.Linear(HIDDEN_SIZE, 16 * 10 * 107)

        self.deconv1 = nn.ConvTranspose2d(16, 32, 3, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 4, 2, 1, 1)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 4, 2, 1, 1)

        self.sig = nn.Sigmoid()
        self.tan = nn.Tanh()
        self.batchNorm16 = nn.BatchNorm2d(16)
        self.batchNorm32 = nn.BatchNorm2d(32)

    def forward(self, x):
        en = self.connv1(x)
        en = self.batchNorm16(en)
        en = self.tan(en)
        en = self.connv2(en)
        en = self.batchNorm32(en)
        en = self.tan(en)
        en = self.connv3(en)
        en = self.batchNorm16(en)
        en = self.tan(en)

        code = self.en_fc(en.view(en.size(0), -1))
        de = self.de_fc(code)

        de = self.deconv1(de.view(de.size(0), 16, 10, 107))
        de = self.batchNorm32(de)
        de = self.tan(de)
        de = self.deconv2(de)
        de = self.batchNorm16(de)
        de = self.tan(de)
        de = self.deconv3(de)
        decoded = self.sig(de)
        return code, decoded


# net = AutoEncoder()
# input = torch.Tensor(64, 1, 43, 431)
# print(input.size())
# code, output = net(input)
# print(output.size())
# summary(net, input_size=(1, 43, 431), batch_size=64, device="cpu")
Loss_list = [6, 5, 4, 4, 4, 3, 2, 2, 2, 2, 1]
Loss_list1 = [6, 5, 4, 4, 4, 3, 2, 2, 2, 2, 1]
# x = range(1, len(Loss_list)+1)
# y = Loss_list
# plt.plot(x, y, '.-')
# plt_title = 'BATCH_SIZE = {}; LEARNING_RATE:{}'.format(64, 0.1)
# plt.title(plt_title)
# plt.xlabel('per10times')
# plt.ylabel('loss')
# plt.savefig("./log/loss.jpg")
# torch.save(net, "./model/deep_auto_encoder_epoch{}_batch{}.pth")
np.save('./model/threshold.npy',1.002561685978435)
threshold = np.load('./model/threshold.npy')
print(threshold)
# loss_set=[]
# loss_f = torch.nn.MSELoss()
# loss=loss_f(output,input)
#
# loss_set.append(loss.detach().numpy())
# stdloss = np.std(loss_set)
# avgloss = np.average(loss_set)
# maxloss = max(loss_set)
# minloss = min(loss_set)
# threshold = avgloss + 3 * stdloss
# print("std:{} avg:{} max:{} min:{} threshold:{}".format(stdloss, avgloss, maxloss, minloss, threshold))
accuracy=99
dataframe = pd.DataFrame({'tag':Loss_list, 'output':Loss_list1,'accuracy(%)':accuracy})
dataframe.to_csv("result.csv",index=False,sep=',')
