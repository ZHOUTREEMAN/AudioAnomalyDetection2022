# -*- coding: utf-8 -*-
# @Description：将每条数据根据excel文件分类并打上标签,并绘制出短时傅里叶变化后的图像以便进行初步的分析，并将用于训练模型的数据集划分好
# @Time : 2022/04/11 14:13
# @Author : XingZhou
# @Email : 329201962@qq.com
import os
import random
import re
import shutil

import librosa
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import data_input_helper
from data_augmentation import to_img2

matplotlib.use('Agg')


def datatrans0407():
    data = pd.read_excel(io=r'./data/mark/mark.xlsx')
    data_raw_list = []
    root_dir = './data/noise'
    data_input_helper.listdir(root_dir, data_raw_list)
    os.mkdir(root_dir + '/1')
    os.mkdir(root_dir + '/2')
    os.mkdir(root_dir + '/3')
    os.mkdir(root_dir + '/pic1')
    os.mkdir(root_dir + '/pic2')
    os.mkdir(root_dir + '/pic3')
    for raw_item in data_raw_list:
        name = re.search("\d+-\d+-\d+-\d+", raw_item).group()
        tag_list = name.split('-')
        it = data[(data['记录点编号'] == int(tag_list[0])) & (data['数据上传时间'] == int(tag_list[2]))]
        if it.values.size != 0:
            # 设置新文件名
            newname = root_dir + '/' + str(it.values[0][5]) + '/' + tag_list[0] + '-' + tag_list[1] + '-' + tag_list[
                2] + '-' + str(it.values[0][5]) + str(it.values[0][6]) + "{:0>2d}".format(
                it.values[0][4]) + "{:0>4d}".format(it.values[0][3]) + '.wav'
            newname2 = root_dir + '/' + 'pic' + str(it.values[0][5]) + '/' + tag_list[0] + '-' + tag_list[1] + '-' + \
                       tag_list[2] + '-' + str(it.values[0][5]) + str(it.values[0][6]) + "{:0>2d}".format(
                it.values[0][4]) + "{:0>4d}".format(it.values[0][3]) + '.png'
            shutil.copy(raw_item, newname)
            y1, sr1 = librosa.load(raw_item, sr=8000)
            target = np.copy(y1)
            img0 = to_img2(target)
            plt.axis("off")
            plt.title('label:' + str(it.values[0][5]))
            plt.imshow(img0)
            plt.savefig(newname2)


def datatrans0418(label_excel, data_root):
    data = pd.read_excel(io=label_excel)
    data_raw_list = []
    root_dir = data_root
    data_input_helper.listdir(root_dir, data_raw_list)
    os.mkdir(root_dir + '/1')
    os.mkdir(root_dir + '/2')
    os.mkdir(root_dir + '/3')
    os.mkdir(root_dir + '/pic1')
    os.mkdir(root_dir + '/pic2')
    os.mkdir(root_dir + '/pic3')
    for raw_item in data_raw_list:
        name = re.search("\d+-\d+-\d+-?\d+", raw_item).group()
        tag_list = name.split('-')
        it = data[data['文件名称'] == name]
        if it.values.size != 0:
            # 设置新文件名
            newname = root_dir + '/' + str(it.values[0][4]) + '/' + tag_list[0] + '-' + tag_list[1] + '-' + tag_list[
                2] + '-' + str(it.values[0][4]) + str(it.values[0][5]) + "{:0>2d}".format(
                it.values[0][6]) + "{:0>4d}".format(it.values[0][7]) + '.wav'
            newname2 = root_dir + '/' + 'pic' + str(it.values[0][4]) + '/' + tag_list[0] + '-' + tag_list[1] + '-' + \
                       tag_list[2] + '-' + str(it.values[0][4]) + str(it.values[0][5]) + "{:0>2d}".format(
                it.values[0][6]) + "{:0>4d}".format(it.values[0][7]) + '.png'
            shutil.copy(raw_item, newname)
            y1, sr1 = librosa.load(raw_item, sr=8000)
            target = np.copy(y1)
            img0 = to_img2(target)
            plt.axis("off")
            plt.title('label:' + str(it.values[0][4]) + 'alarm value:' + str(it.values[0][2]))
            plt.imshow(img0)
            plt.savefig(newname2)


# srcfile 需要复制、移动的文件  dstpath 目的地址
def my_copyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + fname)  # 复制文件


def after_dataset(root, destination):  # 生成after数据集用于网络训练，其中包括三个子集合，分别为训练、测试以及阈值获取,本方法也可以用于新增数据的添加，只需要将输出文件夹定为temp即可。
    data_list_1 = []
    data_list_2 = []
    data_list_3 = []
    data_input_helper.listdir(root + '/1', data_list_1)
    data_input_helper.listdir(root + '/2', data_list_2)
    data_input_helper.listdir(root + '/3', data_list_3)
    random.shuffle(data_list_1)
    for index, data_it in enumerate(data_list_1):
        if index % 10 == 0:
            my_copyfile(data_it, destination + '/test/')
        elif index % 10 == 1 or index % 10 == 2:
            my_copyfile(data_it, destination + '/threshold/')
        else:
            my_copyfile(data_it, destination + '/train/')
    for data_item in data_list_3:
        my_copyfile(data_item, destination + '/test/')
    return


if __name__ == "__main__":
    # datatrans0418(r'./data/mark/label0509.xls', './data/noise0509')
    after_dataset('./data/noise0509', './data/temp')
