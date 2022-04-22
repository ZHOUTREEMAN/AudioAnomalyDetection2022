# -*- coding: utf-8 -*-
# @Description：
# @Author：XingZhou
# @Time：2022/4/22 17:39
# @Email：329201962@qq.com
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset_loaders import  WaterPipeData
from torch.autograd import Variable

BATCH_SIZE = 1
root_dir = 'data/noise_after'
test_dir = 'test'
threshold = np.load('./model/threshold_vae.npy')
test_dataset = WaterPipeData(root_dir, test_dir)
test_loader = DataLoader(test_dataset, BATCH_SIZE)
model = torch.load("./model/noise_deep_vae_encoder_epoch3000_batch64.pth")
data = torch.Tensor(BATCH_SIZE, 28 * 28)
data = Variable(data)
loss_f = torch.nn.MSELoss()
if torch.cuda.is_available():
    net = model.cuda()
    data = data.cuda()
    loss_f = loss_f.cuda()

model.eval()
correct = 0


def check_tag(output, tag):
    if output == tag:
        return 1
    else:
        return 0


tags = []
outputs = []
loss_set = []
for step, (x, tag) in enumerate(test_loader, 1):
    with torch.no_grad():
        x = torch.reshape(x, ((1, 1, 224*224)))
        data.resize_(x.size()).copy_(x)
        decoded , latent, latent_mean, latent_logvar = net(data)
        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        loss = loss_f(decoded, data) + kl_loss
        loss = loss.cpu().detach().numpy()
        loss_set.append(loss)
        output = ""
        if loss > threshold:
            output = "3"
        else:
            output = "1"
        tag = tag[0]
        correct = correct + check_tag(output, tag)
        tags.append(tag)
        outputs.append(output)
        print("output:{}---tag:{}---loss:{}---threshold:{}---{}/{}".format(output, tag, loss, threshold, correct,
                                                                           len(test_dataset)))

accuracy = 100. * correct / len(test_dataset)
print("total accuracy:{:.2f}%".format(accuracy))
dataframe = pd.DataFrame(
    {'tag': tags, 'output': outputs, 'loss': loss_set, 'threshold': threshold, 'accuracy(%)': accuracy})
dataframe.to_csv("./result/result.csv", index=False, sep=',')