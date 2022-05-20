# -*- coding: utf-8 -*-
# @Time : 2021/12/8 19:11
# @Author : XingZhou
# @Email : 329201962@qq.com
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary

from MobileNetV1 import mobilenet_v1
from MobileNetV2 import MobileNetV2

from MobileNetV3 import mobilenetv3_large
from dataset_loaders import WaterPipeDataForFE


def train(model_type, train_loader, BATCH_SIZE, EPOCHS, LR):
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = mobilenet_v1().to(device)
    model2 = MobileNetV2().to(device)
    model3 = mobilenetv3_large().to(device)
    model = {}
    model['v1'] = model1
    model['v2'] = model2
    model['v3'] = model3
    summary(model[model_type], input_size=(3, 224, 224))
    net = model[model_type]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    precision = 0
    data = torch.Tensor(BATCH_SIZE, 28 * 28)
    data = Variable(data)
    if torch.cuda.is_available():
        net = net.cuda()
        data = data.cuda()
    Loss_list_epoch = []
    for epoch in range(EPOCHS):
        Loss_list = []
        net.train()
        for step, (x, y, _) in enumerate(train_loader, 1):
            y = torch.tensor([int(x) for x in y])
            if torch.cuda.is_available():
                y = y.cuda()
            net.zero_grad()
            data.resize_(x.size()).copy_(x)
            out, _ = net(data)
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
        Loss_list_epoch.append(Loss_list[-1])

    x = range(1, len(Loss_list_epoch) + 1)
    y = Loss_list_epoch
    plt.plot(x, y, '.-')
    plt_title = 'EPOCHS = {}; BATCH_SIZE = {}; LEARNING_RATE:{}'.format(EPOCHS, BATCH_SIZE, LR)
    plt.title(plt_title)
    plt.xlabel('per epoch')
    plt.ylabel('loss')
    plt.savefig("./log/loss_FE.jpg")
    print("task over,saving model......")
    torch.save(net, "./model/noise_FE_epoch{}_batch{}.pth".format(EPOCHS, BATCH_SIZE))


def test(test_loader, DATA_SIZE, BATCH_SIZE, EPOCHS):  # 使用测试集进行测试
    net = torch.load("./model/noise_FE_epoch{}_batch{}.pth".format(EPOCHS, BATCH_SIZE))
    data = torch.Tensor(1, 28 * 28)
    data = Variable(data)
    total_correct = 0
    if torch.cuda.is_available():
        net = net.cuda()
        data = data.cuda()
    for step, (x, y, _) in enumerate(test_loader, 1):
        y = torch.tensor([int(x) for x in y])
        if torch.cuda.is_available():
            y = y.cuda()
        data.resize_(x.size()).copy_(x)
        out, _ = net(data)
        pred = torch.argmax(out, dim=1)
        total_correct += torch.eq(y, pred).sum().float().item()
    acc = total_correct / DATA_SIZE
    print('test acc:', acc)
    return


if __name__ == "__main__":
    root_dir = 'data/FE'
    train_dir = 'train'
    test_dir = 'test'
    np.random.seed(123)
    torch.manual_seed(123)
    BATCH_SIZE = 2
    LR = 0.0005
    EPOCHS = 10
    train_dataset = WaterPipeDataForFE(root_dir, train_dir)
    test_dataset = WaterPipeDataForFE(root_dir, test_dir)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, True, drop_last=True)
    test_loader = DataLoader(test_dataset, 1)

    train('v3', train_loader, BATCH_SIZE, EPOCHS, LR)
    test(train_loader, train_dataset.__len__(), BATCH_SIZE, EPOCHS)
