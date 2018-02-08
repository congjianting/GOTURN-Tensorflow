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

实现将跟踪框的位置坐标叠加到对应的图像上显示,用于查阅训练数据标注结果的合理性

# 单个文件夹下的训练标注文件格式如下, 0,1,img_w,img_h,lx,tx,w,h,path
0,1,3968,2976,717,181,2320,1328,/Users/youj/Downloads/siamese-fc-master/ILSVRC15-curation/Data/train/a/000005/000005_0.jpg
0,1,3968,2976,1881,1163,824,410,/Users/youj/Downloads/siamese-fc-master/ILSVRC15-curation/Data/train/a/000005/000005_1.jpg
0,1,3968,2976,2104,1354,355,169,/Users/youj/Downloads/siamese-fc-master/ILSVRC15-curation/Data/train/a/000005/000005_2.jpg
0,1,3968,2976,2961,1847,187,120,/Users/youj/Downloads/siamese-fc-master/ILSVRC15-curation/Data/train/a/000005/000005_3.jpg
0,1,3968,2976,1266,1637,93,70,/Users/youj/Downloads/siamese-fc-master/ILSVRC15-curation/Data/train/a/000005/000005_4.jpg

"""

# 跟踪位置坐标是否归一化
normalize_size  = 1

# 是否需要导出跟踪框显示
need_show_track = 1

# 输入参数
# 训练数据的根路径
input_folder_path    = u'/Volumes/D/track_samples_test'

# 输出参数
# 待查阅的训练数据的根路径
output_folder_path    = u'/Volumes/D/track_samples_test_check'

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

    # 遍历标注清单文件列表
    txt_list = []
    _dir_list(input_folder_path, txt_list, '.txt')

    create_export_folder(output_folder_path)

    # 校验有效标注清单文件对应的图像文件是否均存在,不存在提示用户检查
    for one_txt_path in txt_list:

        one_txt_name    = os.path.basename(one_txt_path)  # txt的文件名称
        img_folder_name = one_txt_name[:-4]               # 对应的图像的文件夹名称

        print img_folder_name

        # 校验txt文件的尺寸,如果等于0,则打印提示
        if os.path.getsize(one_txt_path) == 0:
            print(u"训练标注TXT文件尺寸为空! %s" % one_txt_name)
            continue

        error_txt = 0

        # 打开txt文件,解析出清单中涉及到的图片文件, 要求图片格式均为jpg
        with open(one_txt_path, u"r") as f:

            for one_line in f.readlines():

                one_line   = one_line.strip("\n")

                # 解析出图片文件名称字段
                split_line = one_line.split(",")

                if u"/" in split_line[8]:
                    tmp_paths  = split_line[8].split("/")
                    img_name   = tmp_paths[len(tmp_paths)-1]  # 图片文件名称
                else:
                    img_name   = split_line[8]

                # 校验图片文件是否存在
                if not os.path.exists(os.path.join(input_folder_path, img_folder_name, img_name)):
                    print(u"check image file path error! %s" % img_name)
                    error_txt = 1
                    break

                # 使用其他参数,对跟踪位置进行描绘
                # 读取图像宽度和高度
                img_w = int(split_line[2])
                img_h = int(split_line[3])

                # 根据跟踪坐标是否归一化来计算跟踪框坐标
                if normalize_size == 0:

                    track_lx = int(split_line[4])
                    track_ty = int(split_line[5])
                    track_w  = int(split_line[6])
                    track_h  = int(split_line[7])

                else:

                    track_lx = float(split_line[4])*img_w
                    track_ty = float(split_line[5])*img_h
                    track_w  = float(split_line[6])*img_w
                    track_h  = float(split_line[7])*img_h

                # 读取图片文件
                img = cv2.imread(os.path.join(input_folder_path, img_folder_name, img_name))

                if img is None:
                    print(u"read image file data error! %s" % img_name)
                    sys.exit()
                print img_name

                cv2.rectangle(img, (int(track_lx), int(track_ty)), (int(track_lx+track_w), int(track_ty+track_h)),  (55, 255, 155), 5)

                # 构建出输出路径
                output_img_folder_path = os.path.join(output_folder_path, img_folder_name)
                output_img_path        = os.path.join(output_folder_path, img_folder_name, img_name)

                create_export_folder(output_img_folder_path) # 创建子文件夹路径

                cv2.imwrite(output_img_path, img)

        if error_txt == 1:
            os.remove(one_txt_path)
            print "delete txt path: %s" % one_txt_path


if __name__ == '__main__':

    main()

    print "preprocess show done for checking track train data!"