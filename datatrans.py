# -*- coding: utf-8 -*-
# @Description：将每条数据根据excel文件分类并打上标签
# @Time : 2022/04/11 14:13
# @Author : XingZhou
# @Email : 329201962@qq.com
import os
import re
import shutil
import pandas as pd
import data_input_helper

data = pd.read_excel(io=r'./data/mark/mark.xlsx')
data_raw_list = []
root_dir = './data/noise'
data_input_helper.listdir(root_dir, data_raw_list)
os.mkdir(root_dir + '/1')
os.mkdir(root_dir + '/2')
os.mkdir(root_dir + '/3')
for raw_item in data_raw_list:
    name = re.search("\d+-\d+-\d+-\d+", raw_item).group()
    tag_list = name.split('-')
    it = data[(data['记录点编号'] == int(tag_list[0])) & (data['数据上传时间'] == int(tag_list[2]))]
    if it.values.size != 0:
        # 设置新文件名
        newname = root_dir + '/' + str(it.values[0][5]) + '/' + tag_list[0] + '-' + tag_list[1] + '-' + tag_list[
            2] + '-' + str(it.values[0][5]) + str(it.values[0][6]) + "{:0>2d}".format(
            it.values[0][4]) + "{:0>4d}".format(it.values[0][3]) + '.wav'
        shutil.copy(raw_item, newname)
