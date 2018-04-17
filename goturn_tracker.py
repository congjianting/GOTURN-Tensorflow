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

BATCH_SIZE = 5    # 固定设置成5
WIDTH      = 227
HEIGHT     = 227
scale      = 1.2

# 预测的图片文件序列
predict_dir           = u"/Users/congjt/GOTURN-Tensorflow/debug"

# 图片文件序列的关键词
key_words             = u"debug"

# 训练的模型文件权值路径
model_checkpoint_path = u"./checkpoints/checkpoint.ckpt-146966 "

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

# boxA的格式: 左,上,宽,高, gt
# boxB的格式: 左,上,宽,高, predict
def _compute_iou(boxA, boxB):

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        #iou = interArea / float(boxAArea + boxBArea - interArea)
        iou = interArea / float(boxAArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou

# 计算相似度
def compute_rect_similarity(rect1, rect2, eps):

    delta = (eps * (min(rect1[2]-rect1[0], rect2[2]-rect2[0]) + min(rect1[3]-rect1[1], rect2[3]-rect2[1]))*0.5)

    return ((abs(rect1[0] - rect2[0]) <= delta) and \
           (abs(rect1[1] - rect2[1]) <= delta)  and \
           (abs(rect1[2] - rect2[2]) <= delta)  and \
           (abs(rect1[3] - rect2[3]) <= delta))

# 找出最相似性的矩形位置
def compute_highest_similarity_rect(pre_lx, pre_ty, pre_rx, pre_by, eps):

    rect_num = len(pre_lx)

    # 定义相似的次数
    similar    = np.zeros(rect_num)
    # 定义合并的矩形位置
    merge_rect = []
    for i in range(rect_num):
        merge_rect.append([pre_lx[i], pre_ty[i], pre_rx[i], pre_by[i]])

    for i in range(rect_num):

        for j in range(rect_num):

            # 第i个矩形位置
            rect_i = [pre_lx[i], pre_ty[i], pre_rx[i], pre_by[i]]

            # 第j个矩形位置
            rect_j = [pre_lx[j], pre_ty[j], pre_rx[j], pre_by[j]]

            if compute_rect_similarity(rect_i, rect_j, eps):

                similar[i] += 1

                # 合并merge矩形i和矩形j的位置, 这里说明矩形j可以被合并到i中
                merge_rect[i][0] = (merge_rect[i][0] + rect_j[0]) / 2
                merge_rect[i][1] = (merge_rect[i][1] + rect_j[1]) / 2
                merge_rect[i][2] = (merge_rect[i][2] + rect_j[2]) / 2
                merge_rect[i][3] = (merge_rect[i][3] + rect_j[3]) / 2

    # 找出最大相似次数的索引
    max_idx_  = 0
    max_value = similar[max_idx_]
    for i in range(rect_num):
        if similar[i] > max_value:
            max_idx_ = i
            max_value= similar[i]

    # 返回最大相似度位置
    return merge_rect[max_idx_], max_value*1.0/rect_num

# 待预测图片封装成batchsize的输入序列
def package_input_image_batch(example_img_pad, search_img_pad):

    # 封装模板图像数据
    # 多尺度预测
    # 尺度0
    s0_h = example_img_pad.shape[0]
    s0_w = example_img_pad.shape[1]
    example_im_pad_s0 = example_img_pad  # 0.5
    # 尺度1
    s1_h = int(example_im_pad_s0.shape[0] * 1.2)
    s1_w = int(example_im_pad_s0.shape[1] * 1.2)
    offset = int((s1_h - s0_h) / 2)
    example_im_pad_s1 = np.zeros([s1_h, s1_w, 3], np.uint8)  # 0.6
    example_im_pad_s1[offset:s0_h + offset, offset:s0_w + offset] = example_im_pad_s0
    # 尺度2
    s2_h = int(example_im_pad_s1.shape[0] * 1.2)
    s2_w = int(example_im_pad_s1.shape[1] * 1.2)
    offset = int((s2_h - s1_h) / 2)
    example_im_pad_s2 = np.zeros([s2_h, s2_w, 3], np.uint8)  # 0.72
    example_im_pad_s2[offset:s1_h + offset, offset:s1_w + offset] = example_im_pad_s1
    # 尺度3
    s3_h = int(example_im_pad_s2.shape[0] * 1.2)
    s3_w = int(example_im_pad_s2.shape[1] * 1.2)
    offset = int((s3_h - s2_h) / 2)
    example_im_pad_s3 = np.zeros([s3_h, s3_w, 3], np.uint8)  # 0.864
    example_im_pad_s3[offset:s2_h + offset, offset:s2_w + offset] = example_im_pad_s2
    # 尺度4
    s4_h = int(example_im_pad_s3.shape[0] * 1.2)
    s4_w = int(example_im_pad_s3.shape[1] * 1.2)
    offset = int((s4_h - s3_h) / 2)
    example_im_pad_s4 = np.zeros([s4_h, s4_w, 3], np.uint8)  # 1.0368
    example_im_pad_s4[offset:s3_h + offset, offset:s3_w + offset] = example_im_pad_s3

    # 调试信息
    if False:
        cv2.imwrite("./s0.jpg", example_im_pad_s0)
        cv2.imwrite("./s1.jpg", example_im_pad_s1)
        cv2.imwrite("./s2.jpg", example_im_pad_s2)
        cv2.imwrite("./s3.jpg", example_im_pad_s3)
        cv2.imwrite("./s4.jpg", example_im_pad_s4)

    # resize
    # 尺度0
    example_im_pad_s0 = cv2.resize(example_im_pad_s0, (227, 227), interpolation=cv2.INTER_CUBIC)
    # 尺度1
    example_im_pad_s1 = cv2.resize(example_im_pad_s1, (227, 227), interpolation=cv2.INTER_CUBIC)
    # 尺度2
    example_im_pad_s2 = cv2.resize(example_im_pad_s2, (227, 227), interpolation=cv2.INTER_CUBIC)
    # 尺度3
    example_im_pad_s3 = cv2.resize(example_im_pad_s3, (227, 227), interpolation=cv2.INTER_CUBIC)
    # 尺度4
    example_im_pad_s4 = cv2.resize(example_im_pad_s4, (227, 227), interpolation=cv2.INTER_CUBIC)

    # RGB uint8->float32
    # 尺度0
    example_im_f_s0 = example_im_pad_s0.astype(np.float32)
    # 尺度1
    example_im_f_s1 = example_im_pad_s1.astype(np.float32)
    # 尺度2
    example_im_f_s2 = example_im_pad_s2.astype(np.float32)
    # 尺度3
    example_im_f_s3 = example_im_pad_s3.astype(np.float32)
    # 尺度4
    example_im_f_s4 = example_im_pad_s4.astype(np.float32)

    example_im_f = np.stack([example_im_f_s0, example_im_f_s1, example_im_f_s2, example_im_f_s3, example_im_f_s4])

    # 封装搜索图像数据
    # resize
    search_img_pad = cv2.resize(search_img_pad, (227, 227), interpolation=cv2.INTER_CUBIC)

    # RGB uint8->float32
    search_im_f = search_img_pad.astype(np.float32)

    search_im_f = np.stack([search_im_f, search_im_f, search_im_f, search_im_f, search_im_f])

    return search_im_f, example_im_f

# 解析goturn的batch预测结果
def merge_goturn_predict_rects(fc4, scol_fix, srow_fix, search_im_w, search_im_h):

    # 定义参数
    merge_rect = []
    merge_cred = 0
    eps        = 0.2 # 距离阈值

    if fc4 is None:
        return [], merge_cred

    # batchsize
    batchsize = fc4.shape[0]

    # padsize
    search_padsize = max(search_im_w, search_im_h)

    # 预测的位置坐标需要减去pad的偏移量
    pre_lx = [0.0, 0.0, 0.0, 0.0, 0.0]
    pre_ty = [0.0, 0.0, 0.0, 0.0, 0.0]
    pre_rx = [0.0, 0.0, 0.0, 0.0, 0.0]
    pre_by = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(batchsize):
        pre_lx[i] = (fc4[i][0] / 10 - scol_fix) * search_padsize / search_im_w
        pre_ty[i] = (fc4[i][1] / 10 - srow_fix) * search_padsize / search_im_h
        pre_rx[i] = (fc4[i][2] / 10 - scol_fix) * search_padsize / search_im_w
        pre_by[i] = (fc4[i][3] / 10 - srow_fix) * search_padsize / search_im_h

    # 取出相似度最多的预测位置
    merge_rect, merge_cred = compute_highest_similarity_rect(pre_lx, pre_ty, pre_rx, pre_by, eps)

    # 边界保护 (左,上,右,下)
    track_box    = [0,0,0,0]
    track_box[0] = max(0, merge_rect[0])
    track_box[1] = max(0, merge_rect[1])
    track_box[2] = min(search_im_w-1, merge_rect[2])
    track_box[3] = min(search_im_h-1, merge_rect[3])

    return track_box, merge_cred


# 定义goturn的封装类

# 单尺度跟踪类
# 多尺度跟踪类
class GoturnTracker:

    def __init__(self, model_checkpoint_path, batch_size):

        # 定义参数
        self.batchsize = batch_size

        # 定义跟踪预测结果
        self._predictions_ = {}

        # 创建网络
        self.model_path    = model_checkpoint_path
        self.tracknet      = goturn_net.TRACKNET(self.batchsize, train=False)
        self.tracknet.build()

        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)

        # 初始化
        init       = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        self.sess.run(init)
        self.sess.run(init_local)

        # 加载权值
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.model_path)


    # 定义跟踪方法
    def tracker_evaluation(self, search_im_f, example_im_f):

        # 校验输入参数的size
        if len(search_im_f) != self.batchsize or len(example_im_f) != self.batchsize:
            return None

        [self._predictions_["fc4"]] = self.sess.run([self.tracknet.fc4], \
                                      feed_dict = {self.tracknet.image: search_im_f, self.tracknet.target: example_im_f})

        return self._predictions_["fc4"]


if __name__ == "__main__":

    # 定义命中率统计指标
    hit03  = 0
    hit05  = 0
    hit07  = 0
    all_gt = 0

    # 创建track网络实例
    goturn_tracker = GoturnTracker(model_checkpoint_path, BATCH_SIZE)

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

                # 读取预测的图片对的搜索图片的gt
                gt = []
                if len(txt_list) > 0:
                    with open(txt_list[0], u"r") as fr:
                        gt_line  = fr.readline().strip('\n')
                        gt_line  = fr.readline().strip('\n')  # lx,ty,w,h
                        gt_lines = gt_line.split(',')
                        gt.append(int(gt_lines[0]))
                        gt.append(int(gt_lines[1]))
                        gt.append(int(gt_lines[2]))
                        gt.append(int(gt_lines[3]))


                # 读取模板图片文件
                example_im   = cv2.imread(jpg_list[0])
                example_im   = np.array(example_im[:, :, ::-1])

                # pad
                example_img_h, example_img_w, _     = example_im.shape
                example_padsize                     = max(example_img_h, example_img_w)
                example_img_pad, ecol_fix, erow_fix = pad_image(example_im, [example_padsize,example_padsize])
                ecol_fix                            = ecol_fix*1.0/example_padsize
                erow_fix                            = erow_fix*1.0/example_padsize

                # 读取搜索图片文件
                search_im    = cv2.imread(jpg_list[1])
                search_im    = np.array(search_im[:, :, ::-1])

                # pad
                search_im_h, search_im_w, _        = search_im.shape
                search_padsize                     = max(search_im_h, search_im_w)
                search_img_pad, scol_fix, srow_fix = pad_image(search_im, [search_padsize, search_padsize])
                scol_fix                           = scol_fix * 1.0 / search_padsize
                srow_fix                           = srow_fix * 1.0 / search_padsize

                # 封装待预测图像batch
                search_im_f, example_im_f = package_input_image_batch(example_img_pad, search_img_pad)



                # 单个/多个尺度的图片对预测
                fc4 = goturn_tracker.tracker_evaluation(search_im_f, example_im_f)

                # 定义预测结果
                pre_lx_last = 0
                pre_ty_last = 0
                pre_rx_last = 0
                pre_by_last = 0
                cred        = 0

                merge_rect, merge_cred = merge_goturn_predict_rects(fc4, scol_fix, srow_fix, search_im_w, search_im_h)

                print merge_cred

                if merge_cred > 0:
                    pre_lx_last = merge_rect[0]
                    pre_ty_last = merge_rect[1]
                    pre_rx_last = merge_rect[2]
                    pre_by_last = merge_rect[3]
                    cred        = merge_cred

                # 将当前图片的预测结果写入到预测记录的txt中
                line = "%s/%s/%s,%s/%s/%s,%f,%f,%f,%f\n" % ( key_words,fileName, os.path.basename(jpg_list[0]), \
                                                             key_words, fileName, os.path.basename(jpg_list[1]), \
                                                             pre_lx_last, pre_ty_last, pre_rx_last, pre_by_last )


                fin.write(line)

                # 如果gt存在,则统计预测目标在gt的命中率
                if len(gt) == 4:

                    pre_lx_ = int(pre_lx_last * search_im_w)
                    pre_ty_ = int(pre_ty_last * search_im_h)
                    pre_rx_ = int(pre_rx_last * search_im_w)
                    pre_by_ = int(pre_by_last * search_im_h)

                    union_iou = _compute_iou(gt, [pre_lx_,pre_ty_,pre_rx_-pre_lx_,pre_by_-pre_ty_])

                    all_gt += 1

                    if union_iou > 0.3:
                        hit03 += 1
                    else:
                        print(u"0.3未命中的文件夹名称: %s" % fileName)

                    if union_iou > 0.5:
                        hit05 += 1

                    if union_iou > 0.7:
                        hit07 += 1

                else:

                    print"没有真值标注"

            else:

                print(u"图片文件结构不符合预测格式要求!")
                break

        # 打印出跟踪的命中率指标
        print( u"统计有真值的样本数目: %d" % all_gt)
        print( u"统计03,05,07的样本数目: %d,%d,%d" % (hit03,hit05,hit07))
        print( u"测试样本的命中率: %f,%f,%f" % (hit03*1.0/(all_gt+0.00001), hit05*1.0/(all_gt+0.00001), hit07*1.0/(all_gt+0.00001)) )

