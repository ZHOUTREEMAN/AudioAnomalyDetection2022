# -*- coding: utf-8 -*-
# @Description：
# @Author：XingZhou
# @Time：2022/4/17 15:59
# @Email：329201962@qq.com
import random
import cv2
import librosa
import numpy as np
import pandas as pd

# change wave data to stft
from matplotlib import pyplot as plt


def to_sp(x, n_fft=512, hop_length=256):
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    sp = librosa.amplitude_to_db(np.abs(stft))
    return sp


# add white noise
def add_whitenoise(x, rate=0.002):
    return to_sp(x + rate * np.random.randn(len(x)))


# add pink noise
# https://www.dsprelated.com/showarticle/908.php
def add_pinknoise(x, ncols=11, alpha=0.002):
    """Generates pink noise using the Voss-McCartney algorithm.

    nrows: number of values to generate
    rcols: number of random sources to add

    returns: NumPy array
    """
    nrows = len(x)
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return to_sp(alpha * total.values + x)


# add random line
def draw_line(x, length=[5, 20], thickness_length=[2, 4]):
    result = np.copy(x)
    width = x.shape[1]
    height = x.shape[0]
    angle = [0, np.pi / 2, np.pi, np.pi * 3 / 2]
    np.random.shuffle(angle)

    length = np.random.randint(length[0], length[1])
    x1 = random.randint(length, width - length)
    x2 = x1 + length * np.cos(angle[0])
    y1 = random.randint(length, height - length)
    y2 = y1 + length * np.sin(angle[0])

    thickness = random.randint(thickness_length[0], thickness_length[1])
    color1 = float(np.max(x))

    cv2.line(result, (x1, y1), (int(np.min([width, x2])), int(np.min([height, y2]))), color1, thickness)

    return result


# change Hz to average
def average_Hz(x, length=[2, 4]):
    result = np.copy(x)
    height = x.shape[0]

    length = np.random.randint(length[0], length[1])
    begin = np.random.randint(0, height - length)
    for i in range(length):
        result[begin + i] = np.mean(result[begin + i])

    return result


def normalization(x):
    max_ = 15.6  # np.max(x_train)
    min_ = -74.7  # np.min(x_train)
    mean = 133.4
    sigma = 31.595366

    result = cv2.resize(x, (224, 224))
    result = (result - min_) / (max_ - min_)
    return (result * 255 - mean) / sigma


def to_img2(x):
    result = cv2.resize(to_sp(x), (224, 224))
    return np.array(result)


if __name__ == "__main__":  # transfer
    audio1 = './data/noise/1/1487-867201051535215-20220325101238-13070100.wav'
    y1, sr1 = librosa.load(audio1, sr=8000)
    target = np.copy(y1)
    print(to_sp(target).shape)

    img0 = to_img2(target)
    img0 = normalization(img0)

    img1 = draw_line(img0)

    img2 = add_whitenoise(target)
    img2 = normalization(img2)

    img3 = add_pinknoise(target)
    img3 = normalization(img3)

    img4 = average_Hz(img0)

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 5, 1)
    plt.axis("off")
    plt.title("normal data")
    plt.imshow(img0)

    plt.subplot(1, 5, 2)
    plt.axis("off")
    plt.title("slight line")
    plt.imshow(img1)

    plt.subplot(1, 5, 3)
    plt.axis("off")
    plt.title("add white noise")
    plt.imshow(img2)

    plt.subplot(1, 5, 4)
    plt.axis("off")
    plt.title("add pink noise")
    plt.imshow(img3)

    plt.subplot(1, 5, 5)
    plt.axis("off")
    plt.title("average Hz")
    plt.imshow(img4)
    plt.show()
