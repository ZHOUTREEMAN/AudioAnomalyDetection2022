# -*- coding: utf-8 -*-
# @Description：将每条数据根据excel文件分类并打上标签,并绘制出短时傅里叶变化后的图像以便进行初步的分析
# @Time : 2022/04/11 14:13
# @Author : XingZhou
# @Email : 329201962@qq.com
import os
import re
import shutil

import librosa
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import data_input_helper
import generate_anomaly_sound

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
            img0 = generate_anomaly_sound.to_img2(target)
            plt.axis("off")
            plt.title(it.values[0][5])
            plt.imshow(img0)
            plt.savefig(newname2)


def datatrans0418(label_excel, data_root):
    data = pd.read_excel(io=label_excel)
    data_raw_list = []
    root_dir = data_root
    data_input_helper.listdir(root_dir, data_raw_list)
    # os.mkdir(root_dir + '/1')
    # os.mkdir(root_dir + '/2')
    # os.mkdir(root_dir + '/3')
    # os.mkdir(root_dir + '/pic1')
    # os.mkdir(root_dir + '/pic2')
    # os.mkdir(root_dir + '/pic3')
    for raw_item in data_raw_list:
        print(raw_item)
        name = re.search("\d+-\d+-\d +-?\d+", raw_item).group()
        print(name)
        # tag_list = name.split('-')
        # it = data[data['文件名称'] == name]
        # if it.values.size != 0:
        #     # 设置新文件名
        #     newname = root_dir + '/' + str(it.values[0][5]) + '/' + tag_list[0] + '-' + tag_list[1] + '-' + tag_list[
        #         2] + '-' + str(it.values[0][5]) + str(it.values[0][6]) + "{:0>2d}".format(
        #         it.values[0][4]) + "{:0>4d}".format(it.values[0][3]) + '.wav'
        #     newname2 = root_dir + '/' + 'pic' + str(it.values[0][5]) + '/' + tag_list[0] + '-' + tag_list[1] + '-' + \
        #                tag_list[2] + '-' + str(it.values[0][5]) + str(it.values[0][6]) + "{:0>2d}".format(
        #         it.values[0][4]) + "{:0>4d}".format(it.values[0][3]) + '.png'
        #     shutil.copy(raw_item, newname)
        #     y1, sr1 = librosa.load(raw_item, sr=8000)
        #     target = np.copy(y1)
        #     img0 = generate_anomaly_sound.to_img2(target)
        #     plt.axis("off")
        #     plt.title(it.values[0][5])
        #     plt.imshow(img0)
        #     plt.savefig(newname2)


if __name__ == "__main__":
    datatrans0418(r'./data/mark/label0418.xlsx', './data/noise0418')