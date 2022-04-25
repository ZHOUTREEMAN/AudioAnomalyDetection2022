# -*- coding: utf-8 -*-
# @Description：用于绘图
# @Author：XingZhou
# @Time：2022/4/24 20:16
# @Email：329201962@qq.com
import cv2
import librosa
import sklearn
from matplotlib import pyplot as plt
import librosa.display
from sklearn.preprocessing import StandardScaler
from spafe.features.gfcc import gfcc


def feature_pics():
    sc = StandardScaler()
    audio = './data/noise/3/1483-867201051536148-20220326095837-32010300.wav'
    y, sr = librosa.load(audio, sr=8000)
    stftX = librosa.stft(y, n_fft=512, hop_length=256)
    # 把幅度转成分贝格式
    Xdb = librosa.amplitude_to_db(abs(stftX))
    Xdb_re = cv2.resize(Xdb, (224, 224))
    sc.fit_transform(Xdb_re)
    Xdb_norma=sc.transform(Xdb_re)


    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=44)
    sc.fit_transform(mfccs)
    norm_mfccs=sc.transform(mfccs)

    gfccs = gfcc(sig=y, fs=sr, num_ceps=25)
    sc.fit_transform(gfccs)
    norm_gfccs=sc.transform(gfccs)


    plt.figure(figsize=(7, 7))
    plt.imshow(norm_gfccs)
    plt.show()




if __name__ == "__main__":
    feature_pics()