# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os

import shutil
import cv2
import numpy as np
from copy import deepcopy

""" 函数功能

读取训练清单,粗略统计出目标在搜索图中的中心坐标分布情况,供参考

# goturn的训练清单格式如下, path1,path2,lx,ty,rx,by
train/target/000000_0_0.jpg,train/searching/000000_1_1.jpg,0.375505,0.271421,0.625505,0.521421
train/target/000000_1_2.jpg,train/searching/000000_2_3.jpg,0.358367,0.362525,0.608367,0.612525
train/target/000000_2_4.jpg,train/searching/000000_3_5.jpg,0.214466,0.449596,0.464466,0.699596
train/target/000000_3_6.jpg,train/searching/000000_4_7.jpg,0.261467,0.529234,0.511467,0.779234

"""

# 输入参数
# 训练数据清单文件
input_txt_path        = u'/Users/congjt/data/train_bk_0316/train/train_0319.txt'

# 递归遍历深层目录下的指定扩展名的文件路径列表
def _dir_list(path, allfile, ext):

    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            _dir_list(filepath, allfile, ext)
        else:
            # 拆解文件扩展名
            ext_array = ext.split(u'|')
            for i in ext_array:
                if os.path.splitext(filename)[1] == i.replace(u' ','',5):  # '.meta'
                    allfile.append(filepath)
    return allfile

# 定义主函数
def main():

    # 打开txt文件,解析出清单中涉及到的图片文件, 要求图片格式均为jpg
    with open(input_txt_path, u"r") as f:

        center_x_locations = np.zeros(20)
        center_y_locations = np.zeros(20)

        for one_line in f.readlines():

            one_line   = one_line.strip("\n")

            # 解析出图片文件名称字段
            split_line = one_line.split(",")

            # 根据跟踪坐标是否归一化来计算跟踪框坐标
            track_lx = float(split_line[2])
            track_ty = float(split_line[3])
            track_rx = float(split_line[4])
            track_by = float(split_line[5])

            # 计算出跟踪目标中心位置
            trackcenter_x = (track_lx + track_rx) / 2
            trackcenter_y = (track_ty + track_by) / 2

            # 从X轴方向上统计
            idx_ = min(19,int(trackcenter_x*20))
            center_x_locations[idx_] += 1

            # 从Y轴方向上统计
            idx_ = min(19,int(trackcenter_y*20))
            center_y_locations[idx_] += 1

        # 打印出x和y的分布
        print center_x_locations
        print center_y_locations


if __name__ == '__main__':

    main()

    print "preprocess show done for checking track train data!"