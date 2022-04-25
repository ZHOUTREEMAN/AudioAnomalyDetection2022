# -*- coding: utf-8 -*-
# @Description：获取阈值，并对误差分布进行绘图
# @Author：XingZhou
# @Time：2022/4/22 17:33
# @Email：329201962@qq.com
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset_loaders import WaterPipeData
import seaborn as sns

matplotlib.use('Agg')

root_dir = 'data/noise_after'
train_dir = 'threshold'
np.random.seed(123)
torch.manual_seed(123)
BATCH_SIZE = 1

train_dataset = WaterPipeData(root_dir, train_dir)
train_loader = DataLoader(train_dataset, BATCH_SIZE)

net = torch.load("./model/noise_vae_auto_encoder_epoch20000_batch796.pth")
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
        x = torch.reshape(x, ((1, 1, 224 * 224)))
        data.resize_(x.size()).copy_(x)
        decoded, latent, latent_mean, latent_logvar = net(data)
        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        loss = loss_f(decoded, data) + kl_loss
        loss_set.append(loss.cpu().detach().numpy())
        if step % 10 == 0:
            print("-------------process:{:.0f}%-------------".format(100. * step / len(train_dataset)))

stdloss = np.std(loss_set)
avgloss = np.average(loss_set)
maxloss = max(loss_set)
minloss = min(loss_set)
threshold = avgloss + 3 * stdloss
plt.figure(figsize=(12, 6))
sns.distplot(loss_set, bins=50, hist=True, kde=True, norm_hist=False,
             rug=True, vertical=False, label='normal noise',
             axlabel='loss', rug_kws={'label': 'RUG', 'color': 'b'},
             kde_kws={'label': 'KDE', 'color': 'g', 'linestyle': '--'},
             hist_kws={'color': 'g'})
plt.axvline(threshold, color='r', label='threshold:' + str(threshold))
plt.legend()
plt.savefig('./model/threshold_vae.png')
np.save('./model/threshold_vae.npy', threshold)
print("std:{} avg:{} max:{} min:{} threshold:{}".format(stdloss, avgloss, maxloss, minloss, threshold))
