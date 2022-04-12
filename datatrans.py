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
# print(data)
# print(data[(data['记录点编号']==1483) & (data['数据上传时间']==20220320095836) ])
datarawlist = []
rootdir = './data/noise'
data_input_helper.listdir(rootdir, datarawlist)
os.mkdir(rootdir+'/1')
os.mkdir(rootdir+'/2')
os.mkdir(rootdir+'/3')
for rawitem in datarawlist:
    name = re.search("\d+-\d+-\d+-\d+", rawitem).group()
    taglist=name.split('-')
    it=data[(data['记录点编号'] == int(taglist[0])) & (data['数据上传时间'] == int(taglist[2]))]
    if(it.values.size!=0):
        # 设置新文件名
        newname = rootdir + '/' +str(it.values[0][5])+'/'+ taglist[0]+'-'+ taglist[1]+'-' + taglist[2]+ '-'+str(it.values[0][5])+str(it.values[0][6])+"{:0>2d}".format(it.values[0][4])+"{:0>4d}".format(it.values[0][3])+'.wav'
        shutil.copy(rawitem,newname)
