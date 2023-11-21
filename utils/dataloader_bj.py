# -*- coding: utf-8 -*-
# 表计关键点和数字区域检测
import sys

UTILS_PATH = '..'
sys.path.insert(0, UTILS_PATH)

import math
import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


from utils.utils import cvtColor, preprocess_input

def draw_single_heat_map(pred_hms , save_path="./hm_result"):
    # ------------------------------------- #
    # 单张图片检测结果输入，pred_hms.shape = [classes , height ,width]
    # ------------------------------------- #
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    #pred_hms = pred_hms.cpu().numpy()
    class_id = 0
    for hm in pred_hms:
        max_val = np.max(hm)
        min_val = np.min(hm)
        hm = (hm - min_val)/(max_val-min_val)
        hm *= 255
        cv2.imwrite(os.path.join(save_path,"heat_map_gray_{}.jpg".format(class_id)),hm)
        pred_ = hm.astype(np.uint8)
        pred_heat_map = cv2.applyColorMap(pred_ , cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path,"heat_map_color_{}.jpg".format(class_id)),pred_heat_map)
        class_id += 1


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]  #

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


class CenternetDataset_bj(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(CenternetDataset_bj, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)

        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0] / 4), int(input_shape[1] / 4))
        self.num_classes = num_classes
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self,index):
        index = index % self.length

        # ---------------------------------------------- #
        # 数据增强
        # ---------------------------------------------- #
        #print(self.annotation_lines[index])
        image, point , box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
        #image, point , box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=True)

        # 验证
        #cv2.imwrite(os.path.join("/media/deeppc/data/work_space/guo_workspace/centernet-pytorch-main/yanzhen_image","{}.jpg".format(index)),image)
        #for point_ in point:
        #    image = cv2.circle(image ,(point_[0],point_[1]),3,(255,255,0))
        #for box_ in box:
        #    image = cv2.rectangle(image,(box_[0],box_[1]),(box_[2],box_[3]),(255,0,0))
        #cv2.imwrite(os.path.join("/media/deeppc/data/work_space/guo_workspace/centernet-pytorch-main/yanzhen_image","y_{}.jpg".format(index)),image)

        batch_hm = np.zeros((self.output_shape[0],self.output_shape[1],self.num_classes),dtype=np.float32)
        batch_wh = np.zeros((self.output_shape[0],self.output_shape[1],2),dtype = np.float32)
        batch_reg = np.zeros((self.output_shape[0],self.output_shape[1],2),dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_shape[0],self.output_shape[1]),dtype=np.float32)

        if len(box)!=0:
            boxes = np.array(box[:,:4],dtype=np.float32)
            boxes[:,[0,2]] = np.clip(boxes[:,[0,2]] / self.input_shape[1] * self.output_shape[1] , 0 ,
                                       self.output_shape[1] - 1)
            boxes[:,[1,3]] = np.clip(boxes[:,[1,3]] / self.input_shape[0] * self.output_shape[0] , 0 ,
                                     self.output_shape[0] - 1)
        
        if len(point)!=0:
            points = np.array(point[:,:2] , dtype=np.float32)
            points[:,0] = np.clip(points[:,0] / self.input_shape[1] * self.output_shape[1] , 0 ,
                                       self.output_shape[1] - 1)
            points[:,1] = np.clip(points[:,1] / self.input_shape[0] * self.output_shape[0] , 0 ,
                                       self.output_shape[0] - 1)
        

        for i in range(len(point)):
            point_ = points[i].copy()
            cls_id = int(point[i,-1])

            x , y = point_[0] , point_[1]
            radius = 3
            ct_int = np.array([x,y],dtype=np.float32)
            # 指绘制高斯热力图，其余的均可不进行设置
            batch_hm[:,:,cls_id] = draw_gaussian(batch_hm[:,:,cls_id],ct_int,radius)
            # 先试一下，后序可以看是否需要提高对应
        #hm_temp = batch_hm.transpose(2,0,1)
        #draw_single_heat_map(hm_temp)
        

        for i in range(len(box)):
            bbox = boxes[i].copy()
            cls_id = int(box[i,-1])

            h , w = bbox[3] - bbox[1] , bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h) , math.ceil(w)))
                radius = max(0,int(radius))
                # ------------------------------------------------ #
                # 计算真实框所属的特征点
                # ------------------------------------------------ #
                ct = np.array([(bbox[0]+bbox[2])/2 , (bbox[1]+bbox[3])/2],dtype=np.float32)
                ct_int  = ct.astype(np.int32)
                # ------------------------------------------------ #
                # 绘制高斯热力图
                # ------------------------------------------------ #
                batch_hm[:,:,cls_id] = draw_gaussian(batch_hm[:,:,cls_id],ct_int,radius)
                # ------------------------------------------------ #
                # 计算真实宽高值
                # ------------------------------------------------ #
                batch_wh[ct_int[1] , ct_int[0]] = 1. * w , 1. * h
                # ------------------------------------------------ #
                # 计算中心偏移量
                # ------------------------------------------------ #
                batch_reg[ct_int[1] , ct_int[0]] = ct - ct_int
                # ------------------------------------------------ #
                # 将对应的mask设置为1
                # ------------------------------------------------ #
                batch_reg_mask[ct_int[1] , ct_int[0]] = 1

        image = np.transpose(preprocess_input(image),(2,0,1))

        #hm_temp = batch_hm.transpose(2,0,1)
        #draw_single_heat_map(hm_temp)
        #quit()

        return image , batch_hm , batch_wh , batch_reg , batch_reg_mask


    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=False):
        line = annotation_line.split()
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size   # 使用PIL读取图片格式
        h, w = input_shape    # 使用opencv读取图片后的shape
        # ------------简单说明-------------- #
        #   有图片如下：
        #              1920
        #  1 |-------------------------------|
        #  0 |                               |
        #  8 |                               |
        #  0 |-------------------------------|
        # opencv的imread()读取后： image.shape = (1080,1920,3)
        # PIL的Image.open()读取后： image.size = (1920,1080)

        # ------------------------------#
        #   获得预测框 和 关键点信息
        # ------------------------------#
        point_num = 0
        box_num = 0
        for box_point in line[1:]:
            box_point_ = box_point.split(',')
            if len(box_point_) ==3 :  # 说明是关键点信息
                point_num += 1
            elif len(box_point_) == 5:  # 说明是数据框信息
                box_num += 1
            else:
                raise("Label Txt Error!")


        point = np.zeros((point_num,3),dtype=np.int32)
        box = np.zeros((box_num,5),dtype=np.int32)
        point_i = 0
        box_i = 0
        for box_point in line[1:]:
            box_point_ = box_point.split(',')
            if len(box_point_) == 3:
                for j in range(len(box_point_)):
                    point[point_i][j] = int(box_point_[j])
                point_i += 1
            elif len(box_point_) == 5:
                for j in range(len(box_point_)):
                    box[box_i][j] = int(box_point_[j])
                box_i += 1
            else:
                raise("Label Txt Error!")

        if not random:   # 不使用图像增强, 那就仅仅进行图片的resize ,将关键点信息和预测框进行对应调整即可
            scale = min(w / iw, h / ih)   # 进行原比例缩放，不改变比例
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)  # 随机打乱box的顺序
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            # -------------------------------- #
            #    对关键点进行调整
            # -------------------------------- #
            if len(point) > 0:
                np.random.shuffle(point)
                point[:,0] = point[:,0] * nw / iw + dx
                point[:,1] = point[:,1] * nh / ih + dy
                point[:,0:2][point[:,0:2]<0] = 0
                point[:,0][point[:,0]>w] = w
                point[:,1][point[:,1]>h] = h

            return image_data , point , box

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 从左到右进行翻转

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对关键点和真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        if len(point) > 0:
            np.random.shuffle(point)
            point[:,0] = point[:,0] * nw / iw + dx
            point[:,1] = point[:,1] * nh / ih + dy
            if flip: point[:,0] = w - point[:,0]
            point[:,0:2][point[:,0:2] < 0] = 0
            point[:,0][point[:,0] > w] = w
            point[:,1][point[:,1] > h] = h

        return image_data, point , box


# DataLoader中collate_fn使用
def centernet_dataset_collate_bj(batch):
    imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], []

    for img, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs = np.array(imgs)
    batch_hms = np.array(batch_hms)
    batch_whs = np.array(batch_whs)
    batch_regs = np.array(batch_regs)
    batch_reg_masks = np.array(batch_reg_masks)
    return imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks

if __name__ == "__main__":

    input_shape = [1088,1952]
    train_annotation_path = "../data/data_452_sf6_all/train.txt"
    val_annotation_path = "../data/data_452_sf6_all/test.txt"
    bj_classes = ["center","num","index","r_num"]
    num_classes = len(bj_classes)
    batch_size = 4
    num_workers = 8
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)


    train_dataset   = CenternetDataset_bj(train_lines, input_shape, num_classes, train = True)
    val_dataset     = CenternetDataset_bj(val_lines, input_shape, num_classes, train = False)
    gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=centernet_dataset_collate_bj)
    gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=centernet_dataset_collate_bj)
    
