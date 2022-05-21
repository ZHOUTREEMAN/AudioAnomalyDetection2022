# -*- coding: utf-8 -*-
# @Time : 2021/12/9 17:45
# @Author : XingZhou
# @Email : 329201962@qq.com
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset_loaders import WaterPipeData, WaterPipeDataMfcc
import seaborn as sns

matplotlib.use('Agg')
""" feature:{t-f,mfcc,gfcc,cnn}"""
feature = 't-f'
root_dir = 'data/noise_after'
train_dir = 'threshold'
np.random.seed(123)
torch.manual_seed(123)
BATCH_SIZE = 1

if feature == 't-f':
    input_size1 = 224
    input_size2 = 224
    train_dataset = WaterPipeData(root_dir, train_dir)
elif feature == 'mfcc':
    input_size1 = 44
    input_size2 = 86
    train_dataset = WaterPipeDataMfcc(root_dir, train_dir)

train_loader = DataLoader(train_dataset, BATCH_SIZE)

net = torch.load("./model/noise_{}_deep_auto_encoder_epoch3000_batch64.pth".format(feature))
data = torch.Tensor(BATCH_SIZE, 28 * 28)
data = Variable(data)
loss_f = torch.nn.MSELoss()
if torch.cuda.is_available():
    net = net.cuda()
    data = data.cuda()
    loss_f = loss_f.cuda()

loss_set = []

net.eval()
for step, (x, _, _) in enumerate(train_loader, 1):
    with torch.no_grad():
        x = torch.reshape(x, ((1, 1, input_size1, input_size2)))
        data.resize_(x.size()).copy_(x)
        code, decoded = net(data)
        loss = loss_f(decoded, data)
        loss_set.append(loss.cpu().detach().numpy())
        if step % 10 == 0:
            print("-------------process:{:.0f}%-------------".format(100. * step / len(train_dataset)))

stdloss = np.std(loss_set)
avgloss = np.average(loss_set)
maxloss = max(loss_set)
minloss = min(loss_set)
threshold = avgloss - 2.3 * stdloss
plt.figure(figsize=(12, 6))
sns.distplot(loss_set, bins=50, hist=True, kde=True, norm_hist=False,
             rug=True, vertical=False, label='normal noise',
             axlabel='loss', rug_kws={'label': 'RUG', 'color': 'b'},
             kde_kws={'label': 'KDE', 'color': 'g', 'linestyle': '--'},
             hist_kws={'color': 'g'})
plt.axvline(threshold, color='r', label='threshold:' + str(threshold))
plt.legend()
plt.savefig('./model/threshold_{}_deep.png'.format(feature))
np.save('./model/threshold_{}_deep.npy'.format(feature), threshold)
print("std:{} avg:{} max:{} min:{} threshold:{}".format(stdloss, avgloss, maxloss, minloss, threshold))
