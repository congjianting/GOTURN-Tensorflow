# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import time
import tensorflow as tf
import os
import goturn_net
import cv2
import numpy as np

BATCH_SIZE = 1
WIDTH      = 227
HEIGHT     = 227

# 预测的图片文件序列
predict_dir           = u"/Users/congjt/GOTURN-Tensorflow/test"

# 图片文件序列的关键词
key_words             = u"test"

# 训练的模型文件权值路径
model_checkpoint_path = u"./checkpoints/checkpoint.ckpt-51679"

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

# Pad an image up to the target size
def pad_image(img, target_size):

    rows_missing = target_size[0] - img.shape[0]
    cols_missing = target_size[1] - img.shape[1]

    padded_img   = np.pad(img, ( (int(rows_missing / 2), rows_missing - int(rows_missing / 2)),  # fix bug by cjt
                               (int(cols_missing / 2), cols_missing - int(cols_missing / 2)),
                               (0, 0) ), 'constant' )

    return padded_img, int(cols_missing / 2), int(rows_missing / 2)

if __name__ == "__main__":

    # 创建track网络实例
    tracknet = goturn_net.TRACKNET(BATCH_SIZE, train = False)
    tracknet.build()

    sess       = tf.Session()
    init       = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_local)

    # 加载网络权值
    saver = tf.train.Saver()
    saver.restore(sess, model_checkpoint_path)

    with open("predict_result.txt", "w") as fin:

        # 按照约定格式读取图片文件列表
        lst = os.listdir(predict_dir)
        for fileName in lst:

            # 跳过隐藏文件
            if fileName[0] == ".":
                continue

            # 非图片文件结构
            if '.' not in fileName:

                # 定义jpg的列表
                jpg_list = []
                # 定义txt的列表
                txt_list = []

                # 按照文件名称排序, 名称排在前面的为模板图片, 排在后面的为搜索图片
                _dir_list(os.path.join(predict_dir, fileName), jpg_list, ".jpg")
                _dir_list(os.path.join(predict_dir, fileName), txt_list, ".txt")

                if len(jpg_list)!= 2:
                    print(u"图片文件个数不符合预期2的要求!")
                    break

                # 打印当前预测的图片文件对
                print(u"current predict image pair: %s, %s." % (os.path.basename(jpg_list[0]), os.path.basename(jpg_list[1])))

                # 读取模板图片文件
                example_im   = cv2.imread(jpg_list[0])

                # pad
                example_img_h, example_img_w, _     = example_im.shape
                example_padsize                     = max(example_img_h, example_img_w)
                example_img_pad, ecol_fix, erow_fix = pad_image(example_im, [example_padsize,example_padsize])
                ecol_fix                            = ecol_fix*1.0/example_padsize
                erow_fix                            = erow_fix*1.0/example_padsize

                # resize
                example_img_pad   = cv2.resize(example_img_pad,(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

                # BGR->RGB
                example_im   = np.array(example_img_pad[:, :, ::-1])
                example_im_f = example_im.astype(np.float32)

                example_im_f = np.reshape(example_im_f, [1, example_im_f.shape[0], example_im_f.shape[1], example_im_f.shape[2]])

                # 读取搜索图片文件
                search_im    = cv2.imread(jpg_list[1])

                # pad
                search_im_h, search_im_w, _        = search_im.shape
                search_padsize                     = max(search_im_h, search_im_w)
                search_img_pad, scol_fix, srow_fix = pad_image(search_im, [search_padsize, search_padsize])
                scol_fix                           = scol_fix * 1.0 / search_padsize
                srow_fix                           = srow_fix * 1.0 / search_padsize

                # resize
                search_img_pad = cv2.resize(search_img_pad, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

                # BGR->RGB
                search_im    = np.array(search_img_pad[:, :, ::-1])
                search_im_f  = search_im.astype(np.float32)

                search_im_f  = np.reshape(search_im_f,[1, search_im_f.shape[0], search_im_f.shape[1], search_im_f.shape[2]])

                # 图片对预测
                [fc4] = sess.run([tracknet.fc4], feed_dict={tracknet.image: search_im_f, tracknet.target: example_im_f})

                # 预测的位置坐标需要减去pad的偏移量
                pre_lx = (fc4[0][0] / 10 - scol_fix)*search_padsize/search_im_w
                pre_ty = (fc4[0][1] / 10 - srow_fix)*search_padsize/search_im_h
                pre_rx = (fc4[0][2] / 10 - scol_fix)*search_padsize/search_im_w
                pre_by = (fc4[0][3] / 10 - srow_fix)*search_padsize/search_im_h

                # 将当前图片的预测结果写入到预测记录的txt中
                line = "%s/%s/%s,%s/%s/%s,%f,%f,%f,%f\n" % ( key_words,fileName, os.path.basename(jpg_list[0]), \
                                                             key_words, fileName, os.path.basename(jpg_list[1]), \
                                                             pre_lx, pre_ty, pre_rx, pre_by )


                fin.write(line)

            else:

                print(u"图片文件结构不符合预测格式要求!")
                break

