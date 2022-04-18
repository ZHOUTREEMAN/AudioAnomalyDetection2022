# -*- coding: utf-8 -*-
# @Description：对数据进行增广，特别是异常的音频数据
# @Author：XingZhou
# @Time：2022/4/12 11:01
# @Email：329201962@qq.com
from math import ceil

import librosa
import random
import soundfile
import re
import colorednoise as cn
import numpy as np
import data_input_helper
from matplotlib import pyplot as plt
from itertools import chain

"""
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
"""


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


def get_anomaly():
    old_data_list = []
    root_dir = './data/noise/1'
    target_dir = './data/noise/3'
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


def get_anomaly2():
    old_data_list = []
    root_dir = './data/noise/3'
    data_input_helper.listdir(root_dir, old_data_list)
    for item in old_data_list:
        name = re.search("\d+-\d+-\d+-\d+", item).group()
        tag_list = name.split('-')
        if tag_list[0]=="0000":
            continue
        y, sr = librosa.load(item, sr=8000)
        y_shuffle = sequential_shuffle(y, 5)
        file = root_dir + '/1111-' + tag_list[1] + '-' + tag_list[2] + '-3' + tag_list[3][1:] + '.wav'
        soundfile.write(file, y_shuffle, 8000)
    return


def get_normal():
    old_data_list = []
    root_dir = './data/noise/1'
    data_input_helper.listdir(root_dir, old_data_list)
    for item in old_data_list:
        name = re.search("\d+-\d+-\d+-\d+", item).group()
        tag_list = name.split('-')
        y, sr = librosa.load(item, sr=8000)
        y_shuffle = sequential_shuffle(y, 5)
        file = root_dir + '/1111-' + tag_list[1] + '-' + tag_list[2] + '-1' + tag_list[3][1:] + '.wav'
        soundfile.write(file, y_shuffle, 8000)
    return


# demo_plot()
demo_get_anomaly()
# demo_get_normal()
# get_anomaly()
# get_normal()
# get_anomaly2()