# -*- coding: utf-8 -*-
# @Description：
# @Author：XingZhou
# @Time：2022/4/22 17:39
# @Email：329201962@qq.com
import os

import matplotlib
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from dataset_loaders import WaterPipeData, WaterPipeDataMfcc
from torch.autograd import Variable
import seaborn as sns

matplotlib.use('Agg')

""" feature:{t-f,mfcc,gfcc,cnn}"""
feature = 't-f'
root_dir = 'data/noise_after'
test_dir = 'test'
threshold = np.load('./model/threshold_{}_vae.npy'.format(feature))
if feature == 't-f':
    input_size = 224 * 224
    test_dataset = WaterPipeData(root_dir, test_dir)
elif feature == 'mfcc':
    input_size = 44 * 86
    test_dataset = WaterPipeDataMfcc(root_dir, test_dir)
test_loader = DataLoader(test_dataset, 1)
print(test_dataset.__len__())

model = torch.load("./model/noise_{}_vae_auto_encoder_epoch40000_batch967.pth".format(feature))
data = torch.Tensor(1, 28 * 28)
data = Variable(data)
loss_f = torch.nn.MSELoss()
if torch.cuda.is_available():
    net = model.cuda()
    data = data.cuda()
    loss_f = loss_f.cuda()

model.eval()
correct = 0


def check_tag(output, tag, x, name):
    if output == tag:
        return 1
    else:
        plt.axis("off")
        plt.title(name[0] + ' : ' + tag + '---->' + output)
        plt.imshow(x)
        if not os.path.exists('./result/wrong_pics_{}_vae/'.format(feature)):
            os.makedirs('./result/wrong_pics_{}_vae/'.format(feature))  # 创建路径
        plt.savefig('./result/wrong_pics_{}_vae/'.format(feature) + name[0] + '.png')
        return 0


names = []
tags = []
outputs = []
loss_set = []
loss_1 = []
loss_3 = []
for step, (x, tag, name) in enumerate(test_loader, 1):
    with torch.no_grad():
        raw_x = x[0]
        x = torch.reshape(x, ((1, 1, input_size)))
        data.resize_(x.size()).copy_(x)
        decoded, latent, latent_mean, latent_logvar = net(data)
        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        loss = loss_f(decoded, data) + kl_loss
        loss = loss.cpu().detach().numpy()
        loss_set.append(loss)
        names.append(name[0])
        output = ""
        if loss > threshold:
            output = "3"
        else:
            output = "1"
        tag = tag[0]
        if tag == '1':
            loss_1.append(loss)
        else:
            loss_3.append(loss)
        correct = correct + check_tag(output, tag, raw_x, name)
        tags.append(tag)
        outputs.append(output)
        print("output:{}---tag:{}---loss:{}---threshold:{}---{}/{}".format(output, tag, loss, threshold, correct,
                                                                           len(test_dataset)))

accuracy = 100. * correct / len(test_dataset)
print("total accuracy:{:.2f}%".format(accuracy))
plt.figure(figsize=(12, 6))
sns.distplot(loss_1, bins=50, hist=True, kde=True, norm_hist=False,
             rug=True, vertical=False, label='normal noise',
             axlabel='loss', rug_kws={'label': 'RUG', 'color': 'b'},
             kde_kws={'label': 'KDE', 'color': 'g', 'linestyle': '--'},
             hist_kws={'color': 'g'})
sns.distplot(loss_3, bins=50, hist=True, kde=True, norm_hist=False,
             rug=True, vertical=False, label='anomaly noise',
             axlabel='loss', rug_kws={'label': 'RUG', 'color': 'k'},
             kde_kws={'label': 'KDE', 'color': 'k', 'linestyle': '--'},
             hist_kws={'color': 'k'})
plt.axvline(threshold, color='r', label='threshold:' + str(threshold))
plt.legend()
plt.savefig('./result/result_{}_vae.png'.format(feature))

dataframe = pd.DataFrame(
    {'filename': names, 'tag': tags, 'output': outputs, 'loss': loss_set, 'threshold': threshold,
     'accuracy(%)': accuracy})
dataframe.to_csv("./result/result_{}_vae.csv".format(feature), index=False, sep=',')
