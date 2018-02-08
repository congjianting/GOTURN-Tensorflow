# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os

import shutil
import cv2
import numpy as np
from PIL import Image

""" 函数功能

1.将较大尺寸的训练图像文件(例如1M以上)进行降采样,减小图像的尺寸(会降低图像分辨率).
2.需要更新训练图像的标注结果文件TXT, 使用归一化的跟踪位置坐标进行记录.

"""

# 定义压缩比，数值越大，压缩越小
SIZE_normal     = 1.0
SIZE_small      = 2.0
SIZE_more_small = 4.0

def compress(infile):

    """压缩算法，img.thumbnail对图片进行压缩，还可以改变宽高数值进行压缩"""
    scale = SIZE_normal
    img = Image.open(infile)

    size_of_file = os.path.getsize(infile)
    if size_of_file == 0:
        print("current file size=0 %s" % infile)
        return

    w, h = img.size

    # 根据图像尺寸选择不同的压缩比例
    if max(w,h) > 3200:
        scale = SIZE_more_small
    elif max(w,h) > 1600:
        scale = SIZE_small
    else:
        return

    img.thumbnail((int(w/scale), int(h/scale)))
    img.save(infile)

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
                if os.path.splitext(filename)[1] == i.replace(u' ', '', 5):  # '.meta'
                    allfile.append(filepath)
    return allfile

# 输入参数
# 训练数据的根路径
input_folder_path    = u'/Volumes/E/尤晶晶图片备份/train——原图/e'

# 定义主函数
def main():

    # step1: 降低图片文件的尺寸
    # 遍历输入路径下的所有jpg图片
    jpg_list = []
    _dir_list(input_folder_path, jpg_list, '.jpg')

    for one_jpg in jpg_list:
        compress(one_jpg)

    print "compress jpg done!"

    # step2: 更新训练标注文件TXT的内容
    # 遍历训练数据的标注文件TXT
    txt_list = []
    _dir_list(input_folder_path, txt_list, '.txt')

    for one_txt_path in txt_list:

        one_txt_name    = os.path.basename(one_txt_path)  # txt的文件名称
        img_folder_name = one_txt_name[:-4]               # 对应的图像的文件夹名称

        print img_folder_name

        # 校验txt文件的尺寸,如果等于0,则打印提示
        if os.path.getsize(one_txt_path) == 0:
            print(u"训练标注TXT文件尺寸为空! %s" % one_txt_name)
            continue

        # 创建tmp下的临时文件
        with open(u"/tmp/tmp_track.txt", u"w+") as fout:

            # 打开txt文件,解析出清单中涉及到的图片文件, 要求图片格式均为jpg
            with open(one_txt_path, u"r") as f:

                for one_line in f.readlines():

                    one_line   = one_line.strip("\n")

                    # 解析出图片文件名称字段
                    split_line = one_line.split(",")

                    tmp_paths  = split_line[8].split("/")
                    img_name   = tmp_paths[len(tmp_paths)-1]  # 图片文件名称

                    # 校验图片文件是否存在
                    if not os.path.exists(os.path.join(input_folder_path, img_folder_name, img_name)):
                        print(u"check image file path error! %s" % img_name)
                        sys.exit()

                    # 图片存在,则读取图片的宽度和高度, 进而对文件的标注结果进行更新
                    img          = cv2.imread(os.path.join(input_folder_path, img_folder_name, img_name))

                    if img is None:
                        print(u"read image data error! %s" % img_name)
                        sys.exit()

                    img_h, img_w, _ = img.shape

                    # 得到track box的归一化尺寸坐标
                    if float(split_line[6]) > 1000000:

                        print "large error: %s" % img_folder_name

                    elif float(split_line[6]) > 2:

                        track_lx = int(split_line[4]) * 1.0 / (int(split_line[2]) + 0.000001)
                        track_ty = int(split_line[5]) * 1.0 / (int(split_line[3]) + 0.000001)
                        track_w = int(split_line[6]) * 1.0 / (int(split_line[2]) + 0.000001)
                        track_h = int(split_line[7]) * 1.0 / (int(split_line[3]) + 0.000001)

                    else:

                        track_lx = float(split_line[4])
                        track_ty = float(split_line[5])
                        track_w  = float(split_line[6])
                        track_h  = float(split_line[7])

                    # 写入到临时文件中
                    strresult = "0,1,%d,%d,%f,%f,%f,%f,%s\n" % (img_w, img_h, track_lx, track_ty, track_w, track_h, img_name)
                    fout.write(strresult)

        # 删除掉原TXT文件, 将临时文件拷贝到当前TXT文件的路径下替换
        shutil.move(u"/tmp/tmp_track.txt", one_txt_path)
        print "adjust train txt done!"


if __name__ == '__main__':

    main()

    print "downsample train image data file size!"