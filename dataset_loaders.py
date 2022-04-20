# -*- coding: utf-8 -*-
# @Time : 2021/12/8 16:51
# @Author : XingZhou
# @Email : 329201962@qq.com
import cv2
import sklearn
from torch.utils.data.dataset import Dataset
import os
import data_input_helper
import librosa


class DcaseData(Dataset):
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.path = os.path.join(self.root_dir, self.sub_dir)
        self.data_path = os.listdir(self.path)

    def __getitem__(self, item):
        data_name = self.data_path[item]
        data_item_path = os.path.join(self.root_dir, self.sub_dir, data_name)
        x, sr = data_input_helper.load_clip(data_item_path)
        # 时变频谱图
        # 调用librosa的stft方法可以直接得到短时傅里叶变换的结果
        stftX = librosa.stft(x)
        # 把幅度转成分贝格式
        Xdb = librosa.amplitude_to_db(abs(stftX))
        return Xdb, data_name[:6]

    def __len__(self):
        return len(self.data_path)


class DcaseDataMfcc(Dataset):
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.path = os.path.join(self.root_dir, self.sub_dir)
        self.data_path = os.listdir(self.path)

    def __getitem__(self, item):
        data_name = self.data_path[item]
        data_item_path = os.path.join(self.root_dir, self.sub_dir, data_name)
        mfccs = data_input_helper.extract_feature_mfcc(data_item_path)
        return mfccs, data_name[:6]

    def __len__(self):
        return len(self.data_path)


class DcaseDataGfcc(Dataset):
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.path = os.path.join(self.root_dir, self.sub_dir)
        self.data_path = os.listdir(self.path)

    def __getitem__(self, item):
        data_name = self.data_path[item]
        data_item_path = os.path.join(self.root_dir, self.sub_dir, data_name)
        gfccs = data_input_helper.extract_feature_gfcc(data_item_path)
        return gfccs, data_name[:6]

    def __len__(self):
        return len(self.data_path)


class DaHenData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.data_path = os.listdir(self.path)

    def __getitem__(self, item):
        data_name = self.data_path[item]
        data_item_path = os.path.join(self.root_dir, self.label_dir, data_name)
        x, sr = data_input_helper.load_clip(data_item_path)
        # 时变频谱图
        # 调用librosa的stft方法可以直接得到短时傅里叶变换的结果
        stftX = librosa.stft(x)
        # 把幅度转成分贝格式
        Xdb = librosa.amplitude_to_db(abs(stftX))
        return Xdb, self.label_dir

    def __len__(self):
        return len(self.data_path)


class DaHenDataMfcc(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.data_path = os.listdir(self.path)

    def __getitem__(self, item):
        data_name = self.data_path[item]
        data_item_path = os.path.join(self.root_dir, self.label_dir, data_name)
        mfccs = data_input_helper.extract_feature_mfcc(data_item_path)
        return mfccs, self.label_dir

    def __len__(self):
        return len(self.data_path)


class DaHenDataGfcc(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.data_path = os.listdir(self.path)

    def __getitem__(self, item):
        data_name = self.data_path[item]
        data_item_path = os.path.join(self.root_dir, self.label_dir, data_name)
        gfccs = data_input_helper.extract_feature_gfcc(data_item_path)
        return gfccs, self.label_dir

    def __len__(self):
        return len(self.data_path)


class WaterPipeData(Dataset):  # 直接将原始数据的短时傅里叶变化后直接作为网络输入
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.path = os.path.join(self.root_dir, self.sub_dir)
        self.data_path = os.listdir(self.path)

    def __getitem__(self, item):
        data_name = self.data_path[item]
        data_item_path = os.path.join(self.root_dir, self.sub_dir, data_name)
        x, sr = data_input_helper.load_clip(data_item_path)
        # 时变频谱图
        # 调用librosa的stft方法可以直接得到短时傅里叶变换的结果
        stftX = librosa.stft(x, n_fft=512, hop_length=256)
        # 把幅度转成分贝格式
        Xdb = librosa.amplitude_to_db(abs(stftX))
        Xdb_re = cv2.resize(Xdb, (224, 224))
        Xdb_norma = sklearn.preprocessing.scale(Xdb_re, axis=1)
        return Xdb_norma, data_name[-12]

    def __len__(self):
        return len(self.data_path)


class WaterPipeDataMfcc(Dataset):
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.path = os.path.join(self.root_dir, self.sub_dir)
        self.data_path = os.listdir(self.path)

    def __getitem__(self, item):
        data_name = self.data_path[item]
        data_item_path = os.path.join(self.root_dir, self.sub_dir, data_name)
        mfccs = data_input_helper.extract_feature_mfcc(data_item_path)
        return mfccs, data_name[-12]

    def __len__(self):
        return len(self.data_path)


class WaterPipeDataGfcc(Dataset):
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.path = os.path.join(self.root_dir, self.sub_dir)
        self.data_path = os.listdir(self.path)

    def __getitem__(self, item):
        data_name = self.data_path[item]
        data_item_path = os.path.join(self.root_dir, self.sub_dir, data_name)
        gfccs = data_input_helper.extract_feature_gfcc(data_item_path)
        return gfccs, data_name[-12]

    def __len__(self):
        return len(self.data_path)
