# -*- coding: utf-8 -*-
# @Description：对数据进行增广，特别是正常音频数据
# @Author：XingZhou
# @Time：2022/4/12 11:01
# @Email：329201962@qq.com
import librosa
from matplotlib import pyplot as plt

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


demo_plot()
