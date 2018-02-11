# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import os
import cv2
import numpy as np
from PIL import Image

""" 函数功能

对训练图像文件做PAD操作，将其填0变成正方形，避免训练或预测时数据变形

"""

# 输入参数

# 预测文件的扩展名格式
file_ex             = u'.jpeg | .jpg | .png'

# 是否需要指定输出图像的尺寸，不需要则用None
out_size            = None

# 插值方法
resize_method       = 0  # 0: 三次插值  1: 最近邻插值


# 图像文件的根路径
input_image_rootdir =  u'/Users/congjt/GOTURN-Tensorflow/debug/未命名文件夹'

# 输出参数

# 导出图像文件根路径
output_image_rootdir = input_image_rootdir

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

# 创建文件夹路径
def _create_dst_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# 从一个指定的ndarray中随机裁剪一个区域
def random_crop(image, crop_size):
    height, width = image.shape[1:]
    dy, dx = crop_size
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return image[:, y:(y+dy), x:(x+dx)]

# 定义主函数
def main():

    # 获取文件列表
    file_list = []
    _dir_list(input_image_rootdir, file_list, file_ex)

    # 创建输出目录
    _create_dst_folder(output_image_rootdir)

    # 遍历文件列表，拷贝文件到新路径下
    for one_file in file_list:

        # 读取原文件
        img = cv2.imread(one_file, cv2.IMREAD_UNCHANGED)

        new_with   = max(img.shape[0], img.shape[1])
        new_height = max(img.shape[0], img.shape[1])

        if len(img.shape) > 2:

            emptyImage = np.zeros((new_height, new_with, img.shape[2]), dtype=img.dtype)

            if img.shape[0] > img.shape[1]: # 高度大于宽度时

                dx = int((img.shape[0]-img.shape[1])/2)

                # 宽度补0
                emptyImage[:,dx:img.shape[1]+dx,:] = img[:,:,:]

            else:

                dy = int((img.shape[1]-img.shape[0])/2)

                # 高度补0
                emptyImage[dy:img.shape[0] + dy, :, :] = img[:, :, :]

            # 拆解出标签名称
            split_str = os.path.dirname(one_file).split('/')

            # 判断创建输出子文件路径
            #_create_dst_folder(os.path.join(output_image_rootdir, str(split_str[len(split_str)-1])))

            if out_size is None:
                #cv2.imwrite(os.path.join(output_image_rootdir, str(split_str[len(split_str)-1]), os.path.basename(one_file)), emptyImage)
                cv2.imwrite(
                    os.path.join(output_image_rootdir, os.path.basename(one_file)),
                    emptyImage)
            else:
                if resize_method == 0:
                    res = cv2.resize(emptyImage, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
                else:
                    res = cv2.resize(emptyImage, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
                #cv2.imwrite(os.path.join(output_image_rootdir, str(split_str[len(split_str) - 1]), os.path.basename(one_file)),res)
                cv2.imwrite(
                os.path.join(output_image_rootdir, os.path.basename(one_file)), res)

            # 打印显示规范化后的图像文件
            print( "preprocess padding file: %s" % os.path.basename(one_file) )

        else:

            emptyImage = np.zeros((new_height, new_with), dtype=img.dtype)

            if img.shape[0] > img.shape[1]:  # 高度大于宽度时

                dx = int((img.shape[0] - img.shape[1]) / 2)

                # 宽度补0
                emptyImage[:, dx:img.shape[1] + dx] = img[:, :]

            else:

                dy = int((img.shape[1] - img.shape[0]) / 2)

                # 高度补0
                emptyImage[dy:img.shape[0] + dy, :] = img[:, :]

            # 拆解出标签名称
            split_str = os.path.dirname(one_file).split('/')

            # 判断创建输出子文件路径
            #_create_dst_folder(os.path.join(output_image_rootdir, str(split_str[len(split_str) - 1])))

            if out_size is None:
                #cv2.imwrite(os.path.join(output_image_rootdir, str(split_str[len(split_str) - 1]), os.path.basename(one_file)),emptyImage)
                cv2.imwrite(
                    os.path.join(output_image_rootdir, os.path.basename(one_file)),
                    emptyImage)
            else:
                if resize_method == 0:
                    res = cv2.resize(emptyImage, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
                else:
                    res = cv2.resize(emptyImage, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
                #cv2.imwrite(os.path.join(output_image_rootdir, str(split_str[len(split_str) - 1]), os.path.basename(one_file)),res)
                cv2.imwrite(
                os.path.join(output_image_rootdir, os.path.basename(one_file)), res)

            # 打印显示规范化后的图像文件
            print("preprocess padding file: %s" % os.path.basename(one_file))


if __name__ == '__main__':

    main()

    print "padding image data preprocess done!"
