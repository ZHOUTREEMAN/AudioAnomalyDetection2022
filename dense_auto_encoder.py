# -*- coding: utf-8 -*-
# @Time : 2021/12/13 16:49
# @Author : XingZhou
# @Email : 329201962@qq.com
from torch import nn


class AutoEncoderDense(nn.Module):
    def __init__(self):
        super(AutoEncoderDense, self).__init__()

        self.model = nn.Sequential(
            nn.Linear()
        )

    def forward(self, x):
        decoded = self.model()
        return decoded