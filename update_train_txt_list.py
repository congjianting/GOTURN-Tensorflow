# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os

import shutil
import cv2
import numpy as np

""" 函数功能

将训练清单的图片进行核对, 如果发现训练清单中的图片名称无法匹配,则删除该记录, 只保留查询到的图片文件的记录作为训练清单.

"""

# 定义输入训练清单的路径
input_txt_path    = u"/Users/congjt/data/train_bk_0316/train/train_update_0316.txt"

# 定义参考干净数据集的根路径
input_ref_path    = u"/Users/congjt/data/train_bk_0316_check_okay"

# 定义输出训练清单的路径
output_txt_path   = u"/Users/congjt/GOTURN-Tensorflow/train/train_update_0322.txt"

# 定义输出的垃圾数据的路径
output_trash_path = u"/Users/congjt/data/train_trash_0322"

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

    create_export_folder(output_trash_path)

    # 写入输入路径下的训练清单txt
    with open(output_txt_path, "w") as fin:

        # 读取参考图片文件列表
        ref_jpg_list = []
        _dir_list(input_ref_path, ref_jpg_list, '.jpg')

        # 解析出输入的当前训练清单txt的信息
        with open(input_txt_path, u"r") as f:

            for one_line in f.readlines():

                one_line = one_line.strip("\n")

                # 解析出图片文件名称字段
                split_line = one_line.split(",")

                # 解析出模板图像文件路径
                exampleimg_path = os.path.join("/Users/congjt/data/train_bk_0316",split_line[0])
                # 解析出搜索图像文件路径
                searchimg_path  = os.path.join("/Users/congjt/data/train_bk_0316",split_line[1])

                # 解析出位置坐标
                track_lx = float(split_line[2])
                track_ty = float(split_line[3])
                track_rx = float(split_line[4])
                track_by = float(split_line[5])

                # 对原训练清单中的模板图像文件名称和搜索图片文件名称进行检索匹配
                # 模板图片文件名称
                # train/target/000000.jpg,train/searching/000000.jpg
                tmp_lines        = exampleimg_path.split("/")
                cur_example_name = tmp_lines[len(tmp_lines) - 1]
                tmp_lines        = searchimg_path.split("/")
                cur_search_name  = tmp_lines[len(tmp_lines) - 1]

                bfind1 = 0
                bfind2 = 0

                # 匹配不到则跳过
                for one_jpg in ref_jpg_list:

                    # 参考图片文件名称
                    ref_jpg_name = os.path.basename(one_jpg)

                    if cur_example_name == ref_jpg_name:
                        bfind1 = 1

                    if cur_search_name == ref_jpg_name:
                        bfind2 = 1

                if bfind1 == 0 or bfind2 == 0:

                    print(u"we clear current example image or search image file! %s" % one_line)

                    # 从训练数据集中将这两张图片剪切到另外的数据集下
                    shutil.move(exampleimg_path, os.path.join(output_trash_path, cur_example_name))
                    shutil.move(searchimg_path, os.path.join(output_trash_path, cur_search_name))

                    continue

                # 校验图片文件是否存在(确保这些图片存在)
                if not os.path.exists(exampleimg_path) or not os.path.exists(searchimg_path) :
                    print(u"not find example image or search image file! %s" % one_line)
                    continue

                # 校验obj的归一化坐标, 如果坐标范围不在图像中,则跳过
                if track_lx < 0 and track_rx < 0 and track_ty < 0 and track_by < 0:

                    print(u"obj track < 0! %s" % one_line)

                    # 从训练数据集中将这两张图片剪切到另外的数据集下
                    shutil.move(exampleimg_path, os.path.join(output_trash_path, cur_example_name))
                    shutil.move(searchimg_path, os.path.join(output_trash_path, cur_search_name))

                    continue

                if track_lx > 1 and track_rx > 1 and track_ty > 1 and track_by > 1:
                    print(u"obj track > 1! %s" % one_line)

                    # 从训练数据集中将这两张图片剪切到另外的数据集下
                    shutil.move(exampleimg_path, os.path.join(output_trash_path, cur_example_name))
                    shutil.move(searchimg_path, os.path.join(output_trash_path, cur_search_name))

                    continue


                # 如果存在, 则将该文件的当前行直接写入到更新的训练文本清单中
                fin.write(one_line+"\n")

if __name__ == '__main__':

    main()

    print "update goturn train txt list done!"