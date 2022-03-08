# -*- coding: utf-8 -*-
# @Time : 2021/12/8 19:10
# @Author : XingZhou
# @Email : 329201962@qq.com
from torch import nn

HIDDEN_SIZE = 30


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.en_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )
        self.en_fc = nn.Linear(16 * 10 * 107, HIDDEN_SIZE)
        self.de_fc = nn.Linear(HIDDEN_SIZE, 16 * 10 * 107)
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        en = self.en_conv(x)
        code = self.en_fc(en.view(en.size(0), -1))
        de = self.de_fc(code)
        decoded = self.de_conv(de.view(de.size(0), 16, 10, 107))
        return code, decoded
