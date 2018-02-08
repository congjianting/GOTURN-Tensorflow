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

实现将跟踪框的位置坐标叠加到对应的图像上显示,用于查阅训练数据的坐标合理性

# goturn的训练清单格式如下, path1,path2,lx,ty,rx,by
train/target/000000_0_0.jpg,train/searching/000000_1_1.jpg,0.375505,0.271421,0.625505,0.521421
train/target/000000_1_2.jpg,train/searching/000000_2_3.jpg,0.358367,0.362525,0.608367,0.612525
train/target/000000_2_4.jpg,train/searching/000000_3_5.jpg,0.214466,0.449596,0.464466,0.699596
train/target/000000_3_6.jpg,train/searching/000000_4_7.jpg,0.261467,0.529234,0.511467,0.779234

"""

# 输入参数
# 训练数据清单文件
input_txt_path        = u'../predict_result.txt'

# 训练数据的根路径
input_root            = u"/Users/congjt/GOTURN-Tensorflow"

# 输出参数
# 待查阅的训练数据的根路径
output_folder_path    = u'/Users/congjt/GOTURN-Tensorflow/predict_result_check'

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

# 创建导出文件夹的路径
def create_export_folder(export_folder):
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

# 定义主函数
def main():

    create_export_folder(output_folder_path)

    # 打开txt文件,解析出清单中涉及到的图片文件, 要求图片格式均为jpg
    with open(input_txt_path, u"r") as f:

        for one_line in f.readlines():

            one_line   = one_line.strip("\n")

            # 解析出图片文件名称字段
            split_line = one_line.split(",")

            # 解析出模板图像和搜索图像的相对路径
            exampleimg_name   = split_line[0]
            searchimg_name    = split_line[1]

            # 校验图片文件是否存在
            if not os.path.exists(os.path.join(input_root, exampleimg_name)):
                print(u"check image file path error! %s" % exampleimg_name)
                return
            if not os.path.exists(os.path.join(input_root, searchimg_name)):
                print(u"check image file path error! %s" % searchimg_name)
                return

            # 读取图片文件
            img = cv2.imread(os.path.join(input_root, exampleimg_name))

            if img is None:
                print(u"read image file data error! %s" % exampleimg_name)
                sys.exit()

            img_h, img_w, _ = img.shape

            # 读取图片文件
            img2 = cv2.imread(os.path.join(input_root, searchimg_name))

            if img2 is None:
                print(u"read image file data error! %s" % searchimg_name)
                sys.exit()

            img2_h, img2_w, _ = img2.shape

            # 根据跟踪坐标是否归一化来计算跟踪框坐标
            track_lx = float(split_line[2])*img2_w
            track_ty = float(split_line[3])*img2_h
            track_rx = float(split_line[4])*img2_w
            track_by = float(split_line[5])*img2_h

            # 构建出输出路径
            tmp_paths = split_line[0].split("/")
            img_name  = tmp_paths[len(tmp_paths) - 1]  # 模板图片文件名称
            output_img1_folder_path = os.path.join(output_folder_path, img_name)
            tmp_paths = split_line[1].split("/")
            img_name  = tmp_paths[len(tmp_paths) - 1]  # 搜索图片文件名称
            output_img2_folder_path = os.path.join(output_folder_path, img_name)

            cv2.rectangle(img, (int(img_w/4), int(img_h/4)), (int(img_w*3/4), int(img_h*3/4)), (55, 255, 155), 5)
            cv2.rectangle(img2, (int(track_lx), int(track_ty)), (int(track_rx), int(track_by)), (55, 255, 155), 5)

            cv2.imwrite(output_img1_folder_path, img)
            cv2.imwrite(output_img2_folder_path, img2)


if __name__ == '__main__':

    main()

    print "preprocess show done for checking track train data!"