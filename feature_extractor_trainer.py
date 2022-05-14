# -*- coding: utf-8 -*-
# @Time : 2021/12/8 19:11
# @Author : XingZhou
# @Email : 329201962@qq.com
import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary

from MobileNetV1 import mobilenet_v1
from dataset_loaders import WaterPipeDataForFE


def train(model, train_loader, BATCH_SIZE, EPOCHS, LR, DATA_SIZE):
    net = model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    precision = 0
    data = torch.Tensor(BATCH_SIZE, 28 * 28)
    data = Variable(data)
    if torch.cuda.is_available():
        net = net.cuda()
        data = data.cuda()

    for epoch in range(EPOCHS):
        Loss_list = []
        net.train()
        for step, (x, y, _) in enumerate(train_loader, 1):
            y = torch.tensor([int(x) for x in y])
            if torch.cuda.is_available():
                y = y.cuda()
            net.zero_grad()
            data.resize_(x.size()).copy_(x)
            out = net(data)
            pred = torch.argmax(out, dim=1)
            # precision = y.eq(pred).sum().float().item()
            precision = torch.eq(y, pred).sum().float().item()
            loss = criterion(out, y)  # 计算两者的误差
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            print("----------------epoch:{} step:{} loss:{} acc:{}----------------".format(epoch, step, loss,
                                                                                           precision / BATCH_SIZE))
            Loss_list.append(loss.cpu().detach().numpy())
            precision = 0

    total_correct = 0
    for i in range(EPOCHS):
        for step, (x, y, _) in enumerate(train_loader, 1):
            y = torch.tensor([int(x) for x in y])
            if torch.cuda.is_available():
                y = y.cuda()
            data.resize_(x.size()).copy_(x)
            out = net(data)
            pred = torch.argmax(out, dim=1)
            total_correct += torch.eq(y, pred).sum().float().item()
    acc = total_correct / DATA_SIZE
    print('test acc:', acc)


if __name__ == "__main__":
    root_dir = 'data/FE'
    train_dir = 'train'
    np.random.seed(123)
    torch.manual_seed(123)
    BATCH_SIZE = 2
    LR = 0.0005
    EPOCHS = 100
    train_dataset = WaterPipeDataForFE(root_dir, train_dir)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, True, drop_last=True)
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mobilenet_v1().to(device)
    summary(model, input_size=(3, 224, 224))
    train(model, train_loader, BATCH_SIZE, EPOCHS, LR, train_dataset.__len__())
