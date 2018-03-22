# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os

import shutil
import cv2
import numpy as np
from PIL import Image
import random

""" 函数功能

1.对搜索图像的目标区域叠加随机偏移量,并对叠加随机偏移量后的搜索图像进行crop, 这样可能会导致跟踪目标的box坐标越界.

实现思路:

对待扩增的训练数据,首先判断目标位置框靠近图像的哪个角点,对不同角点附近的目标位置进行叠加随机偏移量,使得目标位置框
更加靠近角点或者有一定比例的越界,从而得到增强后的训练数据.

"""

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

# 创建导出文件夹的路径
def create_export_folder(export_folder):
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

# 输入参数
# 训练数据的根路径
input_folder_path     = "/Volumes/D/0316/b_train"

# 输出参数
# argument data crop
# 训练扩增数据的根路径
output_folder_path    = "/Volumes/D/0316/b_train_crop"

# 定义主函数
def main():

    # 定义计数器
    icounter = 10001

    # 读取输入训练清单文件
    one_txt_path    = os.path.join(input_folder_path, "train/train.txt")  # txt的文件名称

    print("input train txt path: %s" % one_txt_path)

    # 创建输出的训练数据文件夹
    create_export_folder(output_folder_path)
    create_export_folder(os.path.join(output_folder_path, "train"))               # 创建train路径
    create_export_folder(os.path.join(output_folder_path, "train", "searching"))  # 创建train/searching路径
    create_export_folder(os.path.join(output_folder_path, "train", "target"))     # 创建train/target路径

    # 校验txt文件的尺寸,如果等于0,则打印提示
    if os.path.getsize(one_txt_path) == 0:
        print(u"训练标注TXT文件尺寸为空! %s" % one_txt_path)
        sys.exit()

    # 创建输出的训练清单文件
    with open(os.path.join(output_folder_path, "train/train.txt"), u"w+") as fout:

        # 打开txt文件,解析出清单中涉及到的图片文件, 要求图片格式均为jpg
        with open(one_txt_path, u"r") as f:

            # train/target/r180304data0_0 (2)_200000.jpg,train/searching/r180304data0_1 (2)_200001.jpg,0.209375,0.328125,0.718750,0.837500
            for one_line in f.readlines():

                one_line   = one_line.strip("\n")

                # 解析出图片文件名称字段
                split_line = one_line.split(",")

                # 提取出输入图像的target的文件路径
                target_path = os.path.join(input_folder_path, split_line[0])
                # 提取出输入图像的searching的文件路径
                search_path = os.path.join(input_folder_path, split_line[1])

                # 校验图片文件是否存在
                if not os.path.exists(target_path) or not os.path.exists(search_path):
                    print(u"check image file path error! %s" % target_path)
                    continue

                # 模板图片存在,则读取图片的宽度和高度, 进而对文件的标注结果进行更新
                img_target = cv2.imread(target_path)

                if img_target is None:
                    print(u"read image data error! %s" % target_path)
                    sys.exit()

                img_target_h, img_target_w, _ = img_target.shape

                # 搜索图片存在,则读取图片的宽度和高度, 进而对文件的标注结果进行更新
                img_search = cv2.imread(search_path)

                if img_search is None:
                    print(u"read image data error! %s" % target_path)
                    sys.exit()

                img_search_h, img_search_w, _ = img_search.shape

                # 提取target在search图像上的目标位置, 归一化坐标
                track_lx  = float(split_line[2])
                track_ty  = float(split_line[3])
                track_rx  = float(split_line[4])
                track_by  = float(split_line[5])

                # box的中心位置
                trackcenter_x = (track_lx + track_rx) / 2
                trackcenter_y = (track_ty + track_by) / 2

                # box的宽度和高度
                trackbox_w    = track_rx - track_lx
                trackbox_h    = track_by - track_ty

                # 判断当前track的box距离哪个角点更近
                distances = [10000,20000,30000,40000] # 左上, 右上, 右下, 左下

                # 计算左上角到box的中心的距离
                distances[0] = trackcenter_x + trackcenter_y
                # 计算右上角到box的中心的距离
                distances[1] = 1.0 - trackcenter_x + trackcenter_y
                # 计算右下角到box的中心的距离
                distances[2] = 1.0 - trackcenter_x + 1.0 - trackcenter_y
                # 计算左下角到box的中心的距离
                distances[3] = trackcenter_x + 1.0 - trackcenter_y

                min_idx_ = distances.index(min(distances)) # 计算得到最小距离的索引位置

                extra_x_diff = trackbox_w / 10
                extra_y_diff = trackbox_h / 10

                # 根据不同的情况分别做crop操作
                if min_idx_ == 0:

                    # 左上角距离最近
                    w_range = track_lx
                    h_range = track_ty

                    w_offset = w_range/4 + extra_x_diff + random.random()*w_range/2
                    h_offset = h_range/4 + extra_y_diff + random.random()*h_range/2

                    w_offset = max(0.0, w_offset)
                    h_offset = max(0.0, h_offset)

                    # 新的跟踪框的位置==>可能存在负值
                    newtracker_lx = track_lx - w_offset
                    newtracker_ty = track_ty - h_offset
                    newtracker_rx = track_rx - w_offset
                    newtracker_by = track_by - h_offset

                    # 计算出crop的图像区域, 裁剪补0, 保持图像宽度和高度不变
                    w_offset_pixel = int(w_offset * img_search_w)
                    h_offset_pixel = int(h_offset * img_search_h)

                    # 图像crop
                    crop_region = img_search[h_offset_pixel:img_search_h, w_offset_pixel:img_search_w]
                    crop_region_h, crop_region_w, _ = crop_region.shape

                    rows_missing = img_search_h - crop_region_h
                    cols_missing = img_search_w - crop_region_w

                    padded_img   = np.pad(crop_region,
                                         ((int(0), rows_missing),  # fix bug by cjt
                                          (int(0), cols_missing),
                                          (0, 0)), 'constant')
                elif min_idx_ == 1:

                    # 右上角距离最近
                    w_range = 1.0-track_rx
                    h_range = track_ty

                    w_offset = w_range / 4 + extra_x_diff + random.random() * w_range / 2
                    h_offset = h_range / 4 + extra_y_diff + random.random() * h_range / 2

                    w_offset = max(0.0, w_offset)
                    h_offset = max(0.0, h_offset)

                    # 新的跟踪框的位置==>可能存在负值, 或者越界
                    newtracker_lx = track_lx + w_offset
                    newtracker_ty = track_ty - h_offset
                    newtracker_rx = track_rx + w_offset
                    newtracker_by = track_by - h_offset

                    # 计算出crop的图像区域, 裁剪补0, 保持图像宽度和高度不变
                    w_offset_pixel = int(w_offset * img_search_w)
                    h_offset_pixel = int(h_offset * img_search_h)

                    # 图像crop
                    crop_region = img_search[h_offset_pixel:img_search_h, 0:img_search_w-w_offset_pixel]
                    crop_region_h, crop_region_w, _ = crop_region.shape

                    rows_missing = img_search_h - crop_region_h
                    cols_missing = img_search_w - crop_region_w

                    padded_img = np.pad(crop_region,
                                        ((int(0), rows_missing),  # fix bug by cjt
                                         (cols_missing, int(0)),
                                         (0, 0)), 'constant')

                elif min_idx_ == 2:

                    # 右下角距离最近
                    w_range = 1.0 - track_rx
                    h_range = 1.0 - track_by

                    w_offset = w_range / 4 + extra_x_diff + random.random() * w_range / 2
                    h_offset = h_range / 4 + extra_y_diff + random.random() * h_range / 2

                    w_offset = max(0.0, w_offset)
                    h_offset = max(0.0, h_offset)

                    # 新的跟踪框的位置==>可能存在越界
                    newtracker_lx = track_lx + w_offset
                    newtracker_ty = track_ty + h_offset
                    newtracker_rx = track_rx + w_offset
                    newtracker_by = track_by + h_offset

                    # 计算出crop的图像区域, 裁剪补0, 保持图像宽度和高度不变
                    w_offset_pixel = int(w_offset * img_search_w)
                    h_offset_pixel = int(h_offset * img_search_h)

                    # 图像crop
                    crop_region = img_search[0:img_search_h - h_offset_pixel, 0:img_search_w - w_offset_pixel]
                    crop_region_h, crop_region_w, _ = crop_region.shape

                    rows_missing = img_search_h - crop_region_h
                    cols_missing = img_search_w - crop_region_w

                    padded_img = np.pad(crop_region,
                                        ((rows_missing, int(0)),  # fix bug by cjt
                                         (cols_missing, int(0)),
                                         (0, 0)), 'constant')
                elif min_idx_ == 3:

                    # 左下角距离最近
                    w_range = track_lx
                    h_range = 1.0 - track_by

                    w_offset = w_range / 4 + extra_x_diff + random.random() * w_range / 2
                    h_offset = h_range / 4 + extra_y_diff + random.random() * h_range / 2

                    w_offset = max(0.0, w_offset)
                    h_offset = max(0.0, h_offset)

                    # 新的跟踪框的位置==>可能存在负值,或者越界
                    newtracker_lx = track_lx - w_offset
                    newtracker_ty = track_ty + h_offset
                    newtracker_rx = track_rx - w_offset
                    newtracker_by = track_by + h_offset

                    # 计算出crop的图像区域, 裁剪补0, 保持图像宽度和高度不变
                    w_offset_pixel = int(w_offset * img_search_w)
                    h_offset_pixel = int(h_offset * img_search_h)

                    # 图像crop
                    crop_region = img_search[0:img_search_h-h_offset_pixel, w_offset_pixel:img_search_w]
                    crop_region_h, crop_region_w, _ = crop_region.shape

                    rows_missing = img_search_h - crop_region_h
                    cols_missing = img_search_w - crop_region_w

                    padded_img = np.pad(crop_region,
                                        ((rows_missing, int(0)),  # fix bug by cjt
                                         (int(0), cols_missing),
                                         (0, 0)), 'constant')

                else:
                    continue


                # 定义输出图片文件名称的扩展名字
                newname_extra  = "crop%d_" % icounter
                target_newname = newname_extra+os.path.basename(target_path)
                search_newname = newname_extra+os.path.basename(search_path)

                # 将模板图像写入到输出路径下
                cv2.imwrite(os.path.join(output_folder_path, "train", "target", target_newname), img_target)
                # 将搜索图像写入到输出路径下
                cv2.imwrite(os.path.join(output_folder_path, "train", "searching", search_newname), padded_img)

                # 将新的跟踪目标框坐标写入到输出清单中
                line = "train/target/%s,train/searching/%s,%f,%f,%f,%f\n" % (
                    target_newname, search_newname,
                    newtracker_lx, newtracker_ty, newtracker_rx, newtracker_by)
                fout.write(line)
                print(line)

                icounter += 1





if __name__ == '__main__':

    a=random.random()
    b=random.random()

    print a
    print b

    main()

    print "argument crop train image data file!"