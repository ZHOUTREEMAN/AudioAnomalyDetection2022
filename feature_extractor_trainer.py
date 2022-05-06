# -*- coding: utf-8 -*-
# @Time : 2021/12/8 19:11
# @Author : XingZhou
# @Email : 329201962@qq.com
import numpy as np
import torch
from torch import nn, optim

from MobileNetV1 import mobilenet_v1


def train(model, train_XX, train_Y):
    batch_size = 16
    net = model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    train_loss = []
    precision = 0
    for epoch in range(50):

        for i in range(2000 // batch_size):
            x = train_XX[i * batch_size:i * batch_size + batch_size]
            y = train_Y[i * batch_size:i * batch_size + batch_size]

            x = torch.from_numpy(x)  # (batch_size,input_feature_shape)
            y = torch.from_numpy(y)  # (batch_size,label_onehot_shape)
            x = x.cuda()
            y = y.long().cuda()

            out = net(x)

            pred = torch.argmax(out, dim=1)
            precision = y.eq(pred).sum().float().item()
            loss = criterion(out, y)  # 计算两者的误差
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss.append(loss.item())

            print(epoch, i * batch_size, np.mean(train_loss), precision / batch_size)
            train_loss = []
            precision = 0

    total_correct = 0
    for i in range(200):
        x = train_XX[i * 10:i * 10 + 10]
        y = train_Y[i * 10:i * 10 + 10]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = x.cuda()
        y = y.long().cuda()
        out = net(x)
        pred = torch.argmax(out, dim=1)
        total_correct += y.eq(pred).sum().float().item()

    acc = total_correct / 2000.0
    print('test acc:', acc)


if __name__ == "__main__":
    import torch
    from torchsummary import summary

    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mobilenet_v1().to(device)
    summary(model, input_size=(3, 224, 224))

    # # 将梅尔频谱图(灰度图)是转为为3通道RGB图
    # spec_image = cv2.cvtColor(spec_image, cv2.COLOR_GRAY2RGB)
