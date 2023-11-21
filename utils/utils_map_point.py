import glob
import json
import math
import operator
import os
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def get_disctance(point1,point2):
    return math.sqrt( math.pow( point1[0] - point2[0] , 2 ) + math.pow( point1[1] - point2[1] , 2) )

def read_json_file(json_file):
    with open(json_file,'r') as f:
        data_load = json.load(f)
    return data_load

def get_recall(point_truth , point_pred , k = 15):
    recall_count = 0
    for point_t in point_truth:
        for point_p in point_pred:
            if(get_disctance(point_t,point_p)<k):
                recall_count += 1
                continue
    return recall_count

def get_accuracy(point_truth, point_pred , k = 15):
    accuracy_count = 0
    for point_p in point_pred:
        for point_t in point_truth:
            if get_disctance(point_p , point_t) < k:
                accuracy_count += 1
                continue
    return accuracy_count


def get_recall_accuracy(data_load,pred_path):
    bj_classes = ["center","num","index"]

    sum_0 = 0
    sum_1 = 0
    sum_2 = 0

    sum_pred_0 = 0
    sum_pred_1 = 0
    sum_pred_2 = 0

    sum_recall_0 = 0
    sum_recall_1 = 0
    sum_recall_2 = 0

    sum_accuracy_0 = 0
    sum_accuracy_1 = 0
    sum_accuracy_2 = 0

    for i in tqdm(data_load["data"]):
        # 统计各个类别点的数量
        point_truth_0 = i["points"][0]["center"]
        point_truth_1 = i["points"][0]["num"]
        point_truth_2 = i["points"][0]["index"]
        sum_0 += len(point_truth_0)
        sum_1 += len(point_truth_1)
        sum_2 += len(point_truth_2)
        point_pred_0 = []
        point_pred_1 = []
        point_pred_2 = []
        image_id = i["img"].split(".")[0]
        txt_file = os.path.join(pred_path,image_id+".txt")
        if not os.path.exists(txt_file):
            continue
        else:
            with open(txt_file,'r') as f:
                data_pred = f.readlines()
            for line in data_pred:
                line_ = line.split(" ")
                class_id = bj_classes.index(line_[0])
                if class_id == 0:
                    point = [int(line_[1]),int(line_[2])]
                    point_pred_0.append(point)
                elif class_id == 1:
                    point = [int(line_[1]),int(line_[2])]
                    point_pred_1.append(point)
                elif class_id == 2:
                    point = [int(line_[1]),int(line_[2])]
                    point_pred_2.append(point)
                else :
                    raise("ERROR  AAA !!!")
        sum_pred_0 += len(point_pred_0)
        sum_pred_1 += len(point_pred_1)
        sum_pred_2 += len(point_pred_2)

        sum_recall_0 += get_recall(point_truth_0 , point_pred_0)
        sum_recall_1 += get_recall(point_truth_1 , point_pred_1)
        sum_recall_2 += get_recall(point_truth_2 , point_pred_2)

        sum_accuracy_0 += get_accuracy(point_truth_0 , point_pred_0)
        sum_accuracy_1 += get_accuracy(point_truth_1 , point_pred_1)
        sum_accuracy_2 += get_accuracy(point_truth_2 , point_pred_2)


    recall_0 = float(sum_recall_0 / sum_0 )
    recall_1 = float(sum_recall_1 / sum_1 )
    recall_2 = float(sum_recall_2 / sum_2 )

    accuracy_0 = float(sum_accuracy_0 / sum_pred_0)
    accuracy_1 = float(sum_accuracy_1 / sum_pred_1)
    accuracy_2 = float(sum_accuracy_2 / sum_pred_2)

    print("sum_center:{}  sum_sacle:{}  sum_pointer:{}".format(sum_0,sum_1,sum_2))
    print("sum_pre_center:{}  sum_pre_scale:{}  sum_pre_scale:{}".format(sum_pred_0,sum_pred_1,sum_pred_2))
    print("recall_center:{}  recall_scale:{}  recall_pointer:{}".format(recall_0,recall_1,recall_2))
    print("precision_center:{}  precisoin_scale:{}  precision_pointer:{}".format(accuracy_0,accuracy_1,accuracy_2))


if __name__ == "__main__":
    txt_path = "/media/deeppc/data/work_space/guo_workspace/centernet-pytorch-main/map_out/detection-points/"
    test_json = "/media/deeppc/data/work_space/guo_workspace/centernet-pytorch-main/data_bj_num_all/bj/test.json"
    data_load = read_json_file(test_json)
    get_recall_accuracy(data_load,txt_path)
