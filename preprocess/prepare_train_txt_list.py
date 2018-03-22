# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os

import shutil
import cv2
import numpy as np

""" 函数功能

根据训练数据的标注结果,将结果转换成GOTURN模型的训练清单

1. 策略1
根据search图像的obj_box的大小, 将其scale到1/4的w, 此时缩放系数为f1, 将f1应用到对应的target图像, 并对target图像按照新的obj_box进行crop.
由此生成模板图像,搜索图像和goturn的训练清单txt.

2. 策略2 (尽量接近作者的做法)
相比于策略1, 待搜索图像的尺寸相比obj_box仍然为1/2的w, 这样就需要对待搜索图像进行crop.

3. 策略3

# 待验证
根据search图像的obj_box的大小, 将其scale到1/6的w, 此时缩放系数为f1, 将f1应用到对应的target图像, 并对target图像按照新的obj_box进行crop.
由此生成模板图像,搜索图像和goturn的训练清单txt.

"""

# 定义策略的方法
method             = 1

# 定义文件计数器的起始值
ini_counter        = 200000

# 定义输入数据的根路径
input_folder_path  = u'/Volumes/D/0316/b'

# 定义输出数据的根路径
output_folder_path = u'/Volumes/D/0316/b_train'

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

# Pad an image up to the target size
def pad_image(img, target_size):

    rows_missing = target_size[0] - img.shape[0]
    cols_missing = target_size[1] - img.shape[1]

    padded_img   = np.pad(img, ( (int(rows_missing / 2), rows_missing - int(rows_missing / 2)),  # fix bug by cjt
                               (int(cols_missing / 2), cols_missing - int(cols_missing / 2)),
                               (0, 0) ), 'constant' )

    return padded_img, int(cols_missing / 2), int(rows_missing / 2)

# 定义主函数
def main():

    # 采用策略1进行训练数据的准备
    if method == 1:

        # 读取txt的清单
        txt_list = []
        _dir_list(input_folder_path, txt_list, '.txt')

        if len(txt_list) <= 0:
            print "训练数据无记录!"
            return

        # 创建输出文件夹路径
        create_export_folder(output_folder_path)
        create_export_folder(os.path.join(output_folder_path, "train"))
        create_export_folder(os.path.join(output_folder_path, "train", "target"))
        create_export_folder(os.path.join(output_folder_path, "train", "searching"))

        # 写入输入路径下的训练清单txt
        with open(os.path.join(output_folder_path, u"train", "train.txt"), "w") as fin:

            icounter = ini_counter

            # 遍历txt的清单,对输入图像进行变换和crop,同时生成该方法的训练清单
            for one_txt_path in txt_list:

                one_txt_name    = os.path.basename(one_txt_path)  # txt的文件名称
                img_folder_name = one_txt_name[:-4]               # 对应的图像的文件夹名称

                # 校验txt文件的尺寸,如果等于0,则打印提示
                if os.path.getsize(one_txt_path) == 0:
                    print(u"训练标注TXT文件尺寸为空! %s" % one_txt_name)
                    continue

                # 定义当前组的组合记录
                cur_record_list = []

                with open(one_txt_path, u"r") as f:

                    for one_line in f.readlines():

                        # 定义dict
                        cur_img_dic = {}

                        one_line = one_line.strip("\n")

                        # 解析出图片文件名称字段
                        split_line = one_line.split(",")

                        if u"/" in split_line[8]:
                            tmp_paths = split_line[8].split("/")
                            img_name = tmp_paths[len(tmp_paths) - 1]  # 图片文件名称
                        else:
                            img_name = split_line[8]

                        # 校验图片文件是否存在
                        if not os.path.exists(os.path.join(input_folder_path, img_folder_name, img_name)):
                            print(u"check image file path error! %s" % img_name)
                            sys.exit()

                        # 使用其他参数,对跟踪位置进行描绘
                        # 读取图像宽度和高度
                        img_w    = int(split_line[2])
                        img_h    = int(split_line[3])

                        track_lx = float(split_line[4])
                        track_ty = float(split_line[5])
                        track_w  = float(split_line[6])
                        track_h  = float(split_line[7])

                        cur_img_dic["path"]     = os.path.join(input_folder_path, img_folder_name, img_name)
                        cur_img_dic["img_w"]    = img_w
                        cur_img_dic["img_h"]    = img_h
                        cur_img_dic["lx"]       = track_lx
                        cur_img_dic["ty"]       = track_ty
                        cur_img_dic["w"]        = track_w
                        cur_img_dic["h"]        = track_h

                        cur_record_list.append(cur_img_dic) # 将当前的图像文件参数压入到列表中

                # 当前组的样式目前是前一张是模板图,后一张是搜索图, 遇到多个图, 都是前一张是模板,后一张是其搜索图
                for idx_ in range(len(cur_record_list)-1):

                    # 模板图的词典
                    example_dic = cur_record_list[idx_]
                    # 搜索图的词典
                    search_dic  = cur_record_list[idx_+1]

                    # 适当调整idx_+1对应的搜索图的obj_box的位置信息
                    search_img  = cv2.imread(search_dic["path"])
                    example_img = cv2.imread(example_dic["path"])

                    if search_img is None or example_img is None:
                        print "出现读取图像数据错误!"
                        return

                    # 计算出图像的宽度和高度
                    search_img_h, search_img_w, _   = search_img.shape
                    example_img_h, example_img_w, _ = example_img.shape

                    # 对检索图像进行pad, 同时重新计算出跟踪目标的位置
                    search_img_pad, col_fix, row_fix = pad_image(search_img, [max(search_img_h, search_img_w),max(search_img_h, search_img_w)])
                    col_fix = col_fix * 1.0 / max(search_img_h, search_img_w)
                    row_fix = row_fix * 1.0 / max(search_img_h, search_img_w)

                    # 调整搜索图的obj的box, 需要考虑pad图像的偏移量
                    search_cnt_x = (search_dic["lx"]+search_dic["w"]/2)*search_img_w/max(search_img_h, search_img_w) + col_fix
                    search_cnt_y = (search_dic["ty"]+search_dic["h"]/2)*search_img_h/max(search_img_h, search_img_w) + row_fix

                    # box的边界
                    side = max(search_dic["w"]*search_img_w/max(search_img_h, search_img_w), \
                               search_dic["h"]*search_img_h/max(search_img_h, search_img_w))

                    # 统一按照w方向取图
                    f1 = max(1.0, 0.25/side)

                    # 新的box的边界
                    newside = f1*side

                    # 计算出新的box的四个边界, 暂时支持边界越界
                    track_lx = search_cnt_x - newside / 2
                    track_ty = search_cnt_y - newside / 2
                    track_rx = track_lx + newside
                    track_by = track_ty + newside

                    # 定义一个模板图像扩展pad的经验值
                    factor = min(2.2*f1, 3) # 不超过3倍

                    # 对模板图像进行pad, 同时重新计算出跟踪目标的位置
                    example_pad_side                  = int(factor*max(example_img_h, example_img_w))

                    example_img_pad, col_fix, row_fix = pad_image(example_img, [example_pad_side, example_pad_side])
                    col_fix = col_fix * 1.0 / example_pad_side
                    row_fix = row_fix * 1.0 / example_pad_side

                    # 先对模板图像进行crop
                    # 中心位置
                    example_cnt_x = (example_dic["lx"] + example_dic["w"] / 2)*example_img_w/example_pad_side + col_fix
                    example_cnt_y = (example_dic["ty"] + example_dic["h"] / 2)*example_img_h/example_pad_side + row_fix

                    # box的边界
                    example_side    = max(example_dic["w"]*example_img_w/example_pad_side, \
                                          example_dic["h"]*example_img_h/example_pad_side)
                    example_newside = f1*example_side

                    # 计算出新的box的四个边界, 不支持边界越界, 需要保护
                    extrack_lx = example_cnt_x - example_newside / 1
                    extrack_ty = example_cnt_y - example_newside / 1
                    extrack_rx = extrack_lx + example_newside * 2
                    extrack_by = extrack_ty + example_newside * 2

                    # 对异常进行校验,遇到异常,算法人员需要仔细排查和分析设计问题
                    crop_lx = int(extrack_lx * example_pad_side)
                    crop_rx = int(extrack_rx * example_pad_side)
                    crop_ty = int(extrack_ty * example_pad_side)
                    crop_by = int(extrack_by * example_pad_side)

                    if crop_lx < 0 or crop_rx > example_pad_side-1 or crop_ty < 0 or crop_by > example_pad_side-1:
                        print "出现模板图像的CROP区域发生严重错误, 需要排查!"
                        continue

                    example_img_crop = example_img_pad[ crop_ty:crop_by, crop_lx:crop_rx ]

                    example_crop_h, example_crop_w, _ = example_img_crop.shape

                    # 模板图像的PAD尺寸存在严重问题,直接不使用
                    if min(example_crop_w, example_img_h) < 224:
                        print "出现模板图像的尺寸发生严重错误, 需要排查!"
                        continue

                    # 写入到训练下的target路径下
                    tmp_name1 = "%s_%d.jpg" % (os.path.basename(example_dic["path"])[:-4], icounter)
                    icounter = icounter + 1
                    example_img_path = os.path.join(output_folder_path, "train", "target", tmp_name1 )
                    cv2.imwrite(example_img_path, example_img_crop)

                    # 写入到训练下的searching路径下
                    tmp_name2 = "%s_%d.jpg" % (os.path.basename(search_dic["path"])[:-4], icounter)
                    icounter = icounter + 1
                    search_img_path = os.path.join(output_folder_path, "train", "searching", tmp_name2)
                    cv2.imwrite(search_img_path, search_img_pad)

                    # 整理当前模板图像和搜索图像的训练文件路径和跟踪位置坐标到goturn的训练清单中
                    line = "train/target/%s,train/searching/%s,%f,%f,%f,%f\n" % ( tmp_name1,tmp_name2,track_lx,track_ty,track_rx,track_by )
                    fin.write(line)
                    print(line)

    elif method == 3:

        # 读取txt的清单
        txt_list = []
        _dir_list(input_folder_path, txt_list, '.txt')

        if len(txt_list) <= 0:
            print "训练数据无记录!"
            return

        # 读取已经清洗后的训练清单, 这样就有个有效的数据对来源
        example_name_ref_list = []
        search_name_ref_list  = []
        with open("../train/train_update_0207.txt", u"r") as fr:
            # train/target/000000_0_0.jpg,train/searching/000000_1_1.jpg,0.375505,0.271421,0.625505,0.521421
            for one_line in fr.readlines():
                one_line = one_line.strip('\n')
                lines    = one_line.split(",")
                example_name_ref_list.append(os.path.basename(lines[0]))
                search_name_ref_list.append(os.path.basename(lines[1]))


        # 创建输出文件夹路径
        create_export_folder(output_folder_path)
        create_export_folder(os.path.join(output_folder_path, "train"))
        create_export_folder(os.path.join(output_folder_path, "train", "target"))
        create_export_folder(os.path.join(output_folder_path, "train", "searching"))

        # 写入输入路径下的训练清单txt
        with open(os.path.join(output_folder_path, u"train", "train.txt"), "w") as fin:

            icounter = ini_counter

            # 遍历txt的清单,对输入图像进行变换和crop,同时生成该方法的训练清单
            for one_txt_path in txt_list:

                one_txt_name = os.path.basename(one_txt_path)  # txt的文件名称
                img_folder_name = one_txt_name[:-4]  # 对应的图像的文件夹名称

                # 校验txt文件的尺寸,如果等于0,则打印提示
                if os.path.getsize(one_txt_path) == 0:
                    print(u"训练标注TXT文件尺寸为空! %s" % one_txt_name)
                    continue

                # 定义当前组的组合记录
                cur_record_list = []

                with open(one_txt_path, u"r") as f:

                    for one_line in f.readlines():

                        # 定义dict
                        cur_img_dic = {}

                        one_line = one_line.strip("\n")

                        # 解析出图片文件名称字段
                        split_line = one_line.split(",")

                        if u"/" in split_line[8]:
                            tmp_paths = split_line[8].split("/")
                            img_name = tmp_paths[len(tmp_paths) - 1]  # 图片文件名称
                        else:
                            img_name = split_line[8]

                        # 校验图片文件是否存在
                        if not os.path.exists(os.path.join(input_folder_path, img_folder_name, img_name)):
                            print(u"check image file path error! %s" % img_name)
                            sys.exit()

                        # 使用其他参数,对跟踪位置进行描绘
                        # 读取图像宽度和高度
                        img_w = int(split_line[2])
                        img_h = int(split_line[3])

                        track_lx = float(split_line[4])
                        track_ty = float(split_line[5])
                        track_w = float(split_line[6])
                        track_h = float(split_line[7])

                        cur_img_dic["path"] = os.path.join(input_folder_path, img_folder_name, img_name)
                        cur_img_dic["img_w"] = img_w
                        cur_img_dic["img_h"] = img_h
                        cur_img_dic["lx"] = track_lx
                        cur_img_dic["ty"] = track_ty
                        cur_img_dic["w"] = track_w
                        cur_img_dic["h"] = track_h

                        cur_record_list.append(cur_img_dic)  # 将当前的图像文件参数压入到列表中

                # 当前组的样式目前是前一张是模板图,后一张是搜索图, 遇到多个图, 都是前一张是模板,后一张是其搜索图
                for idx_ in range(len(cur_record_list) - 1):

                    # 模板图的词典
                    example_dic = cur_record_list[idx_]
                    # 搜索图的词典
                    search_dic = cur_record_list[idx_ + 1]

                    # 适当调整idx_+1对应的搜索图的obj_box的位置信息
                    search_img = cv2.imread(search_dic["path"])
                    example_img = cv2.imread(example_dic["path"])

                    if search_img is None or example_img is None:
                        print "出现读取图像数据错误!"
                        return

                    # 计算出图像的宽度和高度
                    search_img_h, search_img_w, _ = search_img.shape
                    example_img_h, example_img_w, _ = example_img.shape

                    # 对检索图像进行pad, 同时重新计算出跟踪目标的位置
                    search_img_pad, col_fix, row_fix = pad_image(search_img, [max(search_img_h, search_img_w),
                                                                              max(search_img_h, search_img_w)])
                    col_fix = col_fix * 1.0 / max(search_img_h, search_img_w)
                    row_fix = row_fix * 1.0 / max(search_img_h, search_img_w)

                    # 调整搜索图的obj的box, 需要考虑pad图像的偏移量
                    search_cnt_x = (search_dic["lx"] + search_dic["w"] / 2) * search_img_w / max(search_img_h,
                                                                                                 search_img_w) + col_fix
                    search_cnt_y = (search_dic["ty"] + search_dic["h"] / 2) * search_img_h / max(search_img_h,
                                                                                                 search_img_w) + row_fix

                    # box的边界
                    side = max(search_dic["w"] * search_img_w / max(search_img_h, search_img_w), \
                               search_dic["h"] * search_img_h / max(search_img_h, search_img_w))

                    # 统一按照w方向取图
                    f1 = max(1.0, 0.166 / side)

                    # 新的box的边界
                    newside = f1 * side

                    # 计算出新的box的四个边界, 暂时支持边界越界
                    track_lx = search_cnt_x - newside / 2
                    track_ty = search_cnt_y - newside / 2
                    track_rx = track_lx + newside
                    track_by = track_ty + newside

                    # 定义一个模板图像扩展pad的经验值
                    factor = min(2.2 * f1, 3)  # 不超过3倍

                    # 对模板图像进行pad, 同时重新计算出跟踪目标的位置
                    example_pad_side = int(factor * max(example_img_h, example_img_w))

                    example_img_pad, col_fix, row_fix = pad_image(example_img, [example_pad_side, example_pad_side])
                    col_fix = col_fix * 1.0 / example_pad_side
                    row_fix = row_fix * 1.0 / example_pad_side

                    # 先对模板图像进行crop
                    # 中心位置
                    example_cnt_x = (example_dic["lx"] + example_dic[
                        "w"] / 2) * example_img_w / example_pad_side + col_fix
                    example_cnt_y = (example_dic["ty"] + example_dic[
                        "h"] / 2) * example_img_h / example_pad_side + row_fix

                    # box的边界
                    example_side = max(example_dic["w"] * example_img_w / example_pad_side, \
                                       example_dic["h"] * example_img_h / example_pad_side)
                    example_newside = f1 * example_side

                    # 计算出新的box的四个边界, 不支持边界越界, 需要保护
                    extrack_lx = example_cnt_x - example_newside / 1
                    extrack_ty = example_cnt_y - example_newside / 1
                    extrack_rx = extrack_lx + example_newside * 2
                    extrack_by = extrack_ty + example_newside * 2

                    # 对异常进行校验,遇到异常,算法人员需要仔细排查和分析设计问题
                    crop_lx = int(extrack_lx * example_pad_side)
                    crop_rx = int(extrack_rx * example_pad_side)
                    crop_ty = int(extrack_ty * example_pad_side)
                    crop_by = int(extrack_by * example_pad_side)

                    if crop_lx < 0 or crop_rx > example_pad_side - 1 or crop_ty < 0 or crop_by > example_pad_side - 1:
                        print "出现模板图像的CROP区域发生严重错误, 需要排查!"
                        continue

                    example_img_crop = example_img_pad[crop_ty:crop_by, crop_lx:crop_rx]

                    example_crop_h, example_crop_w, _ = example_img_crop.shape

                    # 模板图像的PAD尺寸存在严重问题,直接不使用
                    if min(example_crop_w, example_img_h) < 224:
                        print "出现模板图像的尺寸发生严重错误, 需要排查!"
                        continue

                    # 校验待写入的图片文件名称是否是干净的有效数据
                    bmatch = False
                    for (ex,se) in zip(example_name_ref_list, search_name_ref_list):

                        if os.path.basename(example_dic["path"])[:-4] in ex and os.path.basename(search_dic["path"])[:-4] in se:
                            bmatch = True
                            break

                    if bmatch == False:
                        print "出现模板图片文件和搜索图片文件不是干净的训练数据情况!"
                        continue

                    # 写入到训练下的target路径下
                    tmp_name1 = "%s_%d.jpg" % (os.path.basename(example_dic["path"])[:-4], icounter)
                    icounter = icounter + 1
                    example_img_path = os.path.join(output_folder_path, "train", "target", tmp_name1)
                    cv2.imwrite(example_img_path, example_img_crop)

                    # 写入到训练下的searching路径下
                    tmp_name2 = "%s_%d.jpg" % (os.path.basename(search_dic["path"])[:-4], icounter)
                    icounter = icounter + 1
                    search_img_path = os.path.join(output_folder_path, "train", "searching", tmp_name2)
                    cv2.imwrite(search_img_path, search_img_pad)

                    # 整理当前模板图像和搜索图像的训练文件路径和跟踪位置坐标到goturn的训练清单中
                    line = "train/target/%s,train/searching/%s,%f,%f,%f,%f\n" % (
                    tmp_name1, tmp_name2, track_lx, track_ty, track_rx, track_by)
                    fin.write(line)
                    print(line)



    else:
        print "do nothing"



if __name__ == '__main__':

    main()

    print "prepare goturn train txt list!"