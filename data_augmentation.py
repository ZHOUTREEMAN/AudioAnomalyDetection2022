# -*- coding: utf-8 -*-
# @Description：对数据进行增广，其中demo开头的函数都是用于绘图直观比较的函数不做批处理工作
# @Author：XingZhou
# @Time：2022/4/12 11:01
# @Email：329201962@qq.com
from math import ceil

import soundfile
import re
import colorednoise as cn
import data_input_helper
from matplotlib import pyplot as plt
from itertools import chain
import random
import cv2
import librosa
import numpy as np
import pandas as pd

"""
原始数据的格式
Format                                   : Wave
File size                                : 85.4 KiB
Duration                                 : 5 s 460 ms
Overall bit rate mode                    : Constant
Overall bit rate                         : 128 kb/s

Audio
Format                                   : PCM
Format settings                          : Little / Signed
Codec ID                                 : 1
Duration                                 : 5 s 460 ms
Bit rate mode                            : Constant
Bit rate                                 : 128 kb/s
Channel(s)                               : 1 channel
Sampling rate                            : 8 000 Hz
Bit depth                                : 16 bits
Stream size                              : 85.3 KiB (100%)

例如：11-865057047734730-20211224115359-12030200.wav
记录点编号：11
数据时间：21年12月24日11时53分59秒
漏损情况 1：无漏
噪声级别 2：2级
水管材料 03：钢塑管
水管口径 0200：200mm

漏损情况：
'1、无漏 2、疑似漏点 3、存在漏点
现场噪声级别：（5级噪声最大）
'1、一级 2、二级 3、三级 4、四级 5、五级
管材：
'1、球墨铸铁管 2、钢管 3、钢塑管 4、PE管 5、PPR管 6、镀锌管 7、铸铁管 8、自应力管(水泥管) '
口径：
40，50，80，100，150.....1800mm
"""
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


def normalization(x):  # 有待改进
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


# 对比不同数据的波形
def demo_plot():
    audio1 = './data/noise/1/1487-867201051535215-20220325101238-13070100.wav'
    audio2 = './data/noise/1/1489-867201051534978-20220324094522-12020150.wav'
    audio3 = './data/noise/2/1487-867201051535215-20220323101236-23070100.wav'
    audio4 = './data/noise/2/1487-867201051535215-20220322101237-23070100.wav'
    audio5 = './data/noise/3/1483-867201051536148-20220326095837-32010300.wav'
    audio6 = './data/noise/3/1483-867201051536148-20220324095837-31010300.wav'

    y1, sr1 = librosa.load(audio1, sr=8000)
    y2, sr2 = librosa.load(audio2, sr=8000)
    y3, sr3 = librosa.load(audio3, sr=8000)
    y4, sr4 = librosa.load(audio4, sr=8000)
    y5, sr5 = librosa.load(audio5, sr=8000)
    y6, sr6 = librosa.load(audio6, sr=8000)

    plt.subplot(321)
    plt.plot(y1)
    plt.title('Normal1')
    plt.axis([0, 50000, -0.5, 0.5])

    plt.subplot(322)
    plt.plot(y2)
    plt.title('Normal2')
    plt.axis([0, 50000, -0.5, 0.5])

    plt.subplot(323)
    plt.plot(y3)
    plt.title('suspected1')
    plt.axis([0, 50000, -0.5, 0.5])

    plt.subplot(324)
    plt.plot(y4)
    plt.title('suspected2')
    plt.axis([0, 50000, -0.5, 0.5])

    plt.subplot(325)
    plt.plot(y5)
    plt.title('abnormal1')
    plt.axis([0, 50000, -0.5, 0.5])

    plt.subplot(326)
    plt.plot(y6)
    plt.title('abnormal2')
    plt.axis([0, 50000, -0.5, 0.5])

    plt.tight_layout()
    plt.show()


def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


def random_enhance(data):
    multiple = random_int_list(0, 5, len(data))
    return np.multiply(multiple, data)


def pink_noise(data):
    beta = 1  # the exponent
    samples = len(data)  # number of samples to generate
    y = cn.powerlaw_psd_gaussian(beta, samples) * 0.1
    return np.add(data, y)


def sequential_shuffle(data, granularity):  # 粗粒度的shuffle 为的就是尽量不破环音频的平顺性,granularity为参与shuffle的切片个数
    section = []
    size = ceil(len(data) / granularity)
    for i in range(0, int(len(data)) + 1, size):
        c = data[i:i + size]
        section.append(c)
    np.random.shuffle(section)
    return list(chain.from_iterable(section))


def demo_get_anomaly():
    audio = './data/noise/1/1487-867201051535215-20220325101238-13070100.wav'
    y, sr = librosa.load(audio, sr=8000)
    y_en = random_enhance(y)
    y_no = pink_noise(y_en)
    plt.subplot(311)
    plt.plot(y)
    plt.title('Original waveform')
    plt.axis([0, 50000, -0.5, 0.5])
    plt.subplot(312)
    plt.plot(y_en)
    plt.title('after random enhancement')
    plt.axis([0, 50000, -0.5, 0.5])
    plt.subplot(313)
    plt.plot(y_no)
    plt.title('after adding pink noise')
    plt.axis([0, 50000, -0.5, 0.5])
    plt.tight_layout()
    plt.show()


def demo_get_normal():
    audio = './data/noise/1/1487-867201051535215-20220325101238-13070100.wav'
    y, sr = librosa.load(audio, sr=8000)
    y_shuffle = sequential_shuffle(y, 5)
    plt.subplot(211)
    plt.plot(y)
    plt.title('Original waveform')
    plt.axis([0, 50000, -0.5, 0.5])
    plt.subplot(212)
    plt.plot(y_shuffle)
    plt.title('after sequential shuffle')
    plt.axis([0, 50000, -0.5, 0.5])
    plt.tight_layout()
    plt.show()
    return


def get_anomaly(root, target):  # 通过先随机增强再加粉噪的方式进行异常数据的生成
    old_data_list = []
    root_dir = root
    target_dir = target
    data_input_helper.listdir(root_dir, old_data_list)
    for item in old_data_list:
        name = re.search("\d+-\d+-\d+-\d+", item).group()
        tag_list = name.split('-')
        y, sr = librosa.load(item, sr=8000)
        y_en = random_enhance(y)
        y_no = pink_noise(y_en)
        file = target_dir + '/0000-' + tag_list[1] + '-' + tag_list[2] + '-3' + tag_list[3][1:] + '.wav'
        soundfile.write(file, y_no, 8000)
    return


def get_anomaly2(root_target):  # 进行粗粒度的shuffle得到的新的异常数据
    old_data_list = []
    root_dir = root_target
    data_input_helper.listdir(root_dir, old_data_list)
    for item in old_data_list:
        name = re.search("\d+-\d+-\d+-\d+", item).group()
        tag_list = name.split('-')
        if tag_list[0] == "0000":
            continue
        y, sr = librosa.load(item, sr=8000)
        y_shuffle = sequential_shuffle(y, 5)
        file = root_dir + '/1111-' + tag_list[1] + '-' + tag_list[2] + '-3' + tag_list[3][1:] + '.wav'
        soundfile.write(file, y_shuffle, 8000)
    return


def get_normal(root_target):  # 进行粗粒度的shuffle得到的新的正常数据
    old_data_list = []
    root_dir = root_target
    data_input_helper.listdir(root_dir, old_data_list)
    for item in old_data_list:
        name = re.search("\d+-\d+-\d+-\d+", item).group()
        tag_list = name.split('-')
        y, sr = librosa.load(item, sr=8000)
        y_shuffle = sequential_shuffle(y, 5)
        file = root_dir + '/1111-' + tag_list[1] + '-' + tag_list[2] + '-1' + tag_list[3][1:] + '.wav'
        soundfile.write(file, y_shuffle, 8000)
    return


def demo_transfer():
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


if __name__ == "__main__":
    get_anomaly2("./data/noise0418/3")
    get_normal("./data/noise0418/1")
