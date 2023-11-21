# -*- coding: utf-8 -*-
# ----------------------------------------------- #
# 将标注好的json文件转化为centernet网络支持输入的txt文件
# 文件转化有两种格式，一种是bj_p:只获得点的位置(用于关键点检测)
# 一种是bj_r:只获得框的位置(用于目标检测，检测刻度数据)，还有就是
# bj_pr :关键点以及框的位置都获得(同时进行关键点检测以及数字区域检测)
# ----------------------------------------------- #

import os
import random
import json
import argparse
import cv2
from PIL import Image

#bj_classes = ['index','num','center','f_num','r_num','r_fnum']
#bj_classes = ['index','num','center','r_num','r_fnum']
#bj_classes = ['index','num','center','f_num']
#bj_classes = ['r_num','r_fnum'] # 只有框,以后都按照这种格式标注
#bj_classes = ['center' , 'num' , 'index' , 'r_num' , 'r_fnum']
bj_classes = ['center','num','index','r_num']

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to change annotation')
    parser.add_argument('--mode',default='bj_pr',help='bj_pr | bj_p | bj_r')
    parser.add_argument('--image_dir',default='./data_bj_no_blqnum/bj/imgs/',help='the image path')
    parser.add_argument('--train_json',default='./data_bj_no_blqnum/bj/train.json',help='the train json path')
    parser.add_argument('--test_json',default='./data_bj_no_blqnum/bj/test.json',help='the test json path')
    parser.add_argument('--save_path',default='./data_bj_no_blqnum/',help='the train txt save path')

    args = parser.parse_args()
    return args

def readJsonWriteTxt(jsonfile,save_base_path,txtfile,image_dir,mode):
    """ 读取json文件，将其转化为txt文件格式"""
    with open(jsonfile , 'r') as f:
        json_data = json.load(f)
    json_len = len(json_data['data'])
    ftxt = open(os.path.join(save_base_path,txtfile),'w')
    if mode == 'bj_pr':
        for i in json_data['data']:
            imagename = i['img']
            image_path = os.path.join(image_dir,imagename)
            ftxt.write(image_path)
            annotation_p = i['points'][0]
            for label_name in bj_classes:
                if label_name.startswith('r_'):
                    continue
                points = annotation_p[label_name]
                for point in points:
                    x = int(point[0])
                    y = int(point[1])
                    label = bj_classes.index(label_name)
                    ftxt.write(' {},{},{}'.format(x,y, label))
            annotation_r = i['rectangles'][0]
            for label_name in bj_classes:
                if not label_name.startswith('r_'):
                    continue
                rectangles = annotation_r[label_name]
                for rectangle in rectangles:
                    x1 , y1 = int(rectangle[0][0]) , int(rectangle[0][1])
                    x2 , y2 = int(rectangle[1][0]) , int(rectangle[1][1])
                    label = bj_classes.index(label_name)
                    ftxt.write(' {},{},{},{},{}'.format(x1, y1, x2, y2, label))
            ftxt.write('\n')

    elif mode == 'bj_p':
        for i in json_data['data']:
            imagename = i['img']
            image_path = os.path.join(image_dir,imagename)
            ftxt.write(image_path)
            annotation_p = i['points'][0]
            for label_name in bj_classes:
                if label_name.startswith('r_'):
                    continue
                points = annotation_p[label_name]
                for point in points:
                    x = int(point[0])
                    y = int(point[1])
                    label = bj_classes.index(label_name)
                    ftxt.write(' {},{},{}'.format(x,y,label))
                ftxt.write('\n')

    elif mode == 'bj_r':
        for i in json_data['data']:
            imagename = i['img']
            image_path = os.path.join(image_dir, imagename)
            ftxt.write(image_path)
            annotation_r = i['rectangles'][0]
            count = 0
            for label_name in bj_classes:
                if not label_name.startswith('r_'):
                    continue
                rectangles = annotation_r[label_name]
                for rectangle in rectangles:
                    x1, y1 = int(rectangle[0][0]), int(rectangle[0][1])
                    x2, y2 = int(rectangle[1][0]), int(rectangle[1][1])
                    label = count
                    ftxt.write(' {},{},{},{},{}'.format(x1, y1, x2, y2, label))
                count += 1
            ftxt.write('\n')
    else:
        raise Exception('args.mode:{} is not exits'.format(mode))
    ftxt.close()

if __name__ == '__main__':
    args = get_parser()
    mode = args.mode
    train_json = args.train_json
    test_json = args.test_json
    save_base_path = args.save_path
    image_dir = args.image_dir
    readJsonWriteTxt(train_json, save_base_path,'train.txt', image_dir, mode)
    readJsonWriteTxt(test_json,save_base_path,'test.txt',image_dir,mode)