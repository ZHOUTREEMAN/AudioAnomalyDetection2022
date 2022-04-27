# -*- coding: utf-8 -*-
# @Description：
# @Author：XingZhou
# @Time：2022/4/27 16:12
# @Email：329201962@qq.com
import torch.nn as nn


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 640,640,3 -&gt; 320,320,32
            conv_bn(3, 32, 2),
            # 320,320,32 -&gt; 320,320,64
            conv_dw(32, 64, 1),

            # 320,320,64 -&gt; 160,160,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 160,160,128 -&gt; 80,80,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        # 80,80,256 -&gt; 40,40,512
        self.stage2 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        # 40,40,512 -&gt; 20,20,1024
        self.stage3 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def mobilenet_v1(pretrained=False, progress=True):
    model = MobileNetV1()
    if pretrained:
        print("mobilenet_v1 has no pretrained model")
    return model


if __name__ == "__main__":
    import torch
    from torchsummary import summary

    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mobilenet_v1().to(device)
    summary(model, input_size=(3, 416, 416))