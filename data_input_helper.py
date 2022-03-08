# -*- coding: utf-8 -*-
# @Time : 2021/12/8 16:53
# @Author : XingZhou
# @Email : 329201962@qq.com

import numpy as np
import os
# audio
import librosa
import librosa.display
# normalization
import sklearn
from spafe.features.gfcc import gfcc


# 获取文件夹中所有文件的路径传入存储的list


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


# 用于读取音频片段，库里的片段几乎都是10s，若小于10秒，将它们补零。采样率11025，10秒一共110250个采样点。
def load_clip(filename):
    x, sr = librosa.load(filename)
    # x = np.pad(x,(0,110250-x.shape[0]),'constant')
    x = np.array(x).astype(float)
    return x, sr


# 提取片段的mfcc并进行normalization。
def extract_feature_mfcc(filename):
    x, sr = load_clip(filename)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=43)
    norm_mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    return norm_mfccs


# 提取片段的gfcc并进行normalization。
def extract_feature_gfcc(filename):
    # read wave file
    x, sr = load_clip(filename)
    # compute gfccs
    gfccs = gfcc(sig=x, fs=sr, num_ceps=64)
    norm_gfccs = sklearn.preprocessing.scale(gfccs, axis=1)
    return norm_gfccs


def get_features_mfcc(filenames):
    features = np.empty((0, 40, 431))
    for filename in (filenames):
        mfccs = extract_feature_mfcc(filename)
        features = np.append(features, mfccs[None], axis=0)
    return np.array(features)


def get_features_gfcc(filenames):
    features = np.empty((0, 1000, 13))
    for filename in (filenames):
        gfccs = extract_feature_gfcc(filename)
        features = np.append(features, gfccs[None], axis=0)
    return np.array(features)


def get_data_set_mfcc(filenames, filename):
    data = []
    with open(filename, "r") as f:
        for filename in (filenames):
            labels = int(f.readline())
            mfccs = extract_feature_mfcc(filename)
            data_item = [mfccs, labels]
            data.append(data_item)
    return np.array(data)


def get_labels(filename):
    labels = []
    with open(filename, "r") as f:
        for line in f:  # 遍历每一行
            number = int(line)
            labels.append(number)
    f.close()
    return np.array(labels, dtype=np.int)
