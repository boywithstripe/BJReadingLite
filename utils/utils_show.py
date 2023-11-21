# -*- coding: utf-8 -*-
import cv2
import torch
import os
import plt

def draw_single_heat_map(pred_hms , save_path="./hm_result"):
    # ------------------------------------- #
    # 单张图片检测结果输入，pred_hms.shape = [classes , height ,width]
    # ------------------------------------- #
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    pred_hms = pred_hms.cpu().numpy()
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


def draw_3D_heat_map(pred_hms , save_path="./"):
    pred_hms = pred_hms.cpu().numpy()

    return 0
