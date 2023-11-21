# -*- coding: utf-8 -*-
# ------------------------------------------------------- #
# 对检测到的关键点进行读数
# ------------------------------------------------------- #
from email.encoders import encode_noop
import os
import json
from re import S
import numpy as np
import math
import cv2
from scipy.optimize import leastsq
from PIL import ImageDraw, ImageFont


def get_center(p1,p2):
    p = []
    p.append((p1[0]+p2[0])/2)
    p.append((p1[1]+p2[1])/2)
    return p

def residuals(p, d):
    a, b, r = p
    return r ** 2 - (d[:, 0] - a) ** 2 - (d[:, 1] - b) ** 2

def get_distence(p1,p2):
    return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))

def encoding(p):
    str1 = str(round(float(p[0]),4))
    str2 = str(round(float(p[1]),4))
    s = str1 + "p" +str2
    return s
  
def decoding(s):
    idx = s.find('p')
    str1 = s[:idx]
    str2 = s[idx+1:]
    p = [float(str1),float(str2)]
    return p

def outlierJudge(point,points,threshold):
    return False


def get_startpoint(scale_angle,pointer_angle,scaledict):
    idx1 = 0
    idx2 = len(scale_angle) - 1
    angle_diff = scale_angle[idx1] + 360 - scale_angle[idx2]

    for i in range(len(scale_angle)-1):
        if abs(scale_angle[i+1]-scale_angle[i]) > angle_diff:
            idx1 = i
            idx2 = i + 1
            angle_diff = abs(scale_angle[i+1]-scale_angle[i])

    if idx1 != 0 or idx2 != (len(scale_angle) - 1):  # 找到起始点下标 idx2
        temp_angle = 360 - scale_angle[idx2]
        angle = [i for i in scale_angle]
        count = 0
        tempdict = dict()
        while count < len(scale_angle):
            idx = scaledict[scale_angle[count]]
            scale_angle[count] = angle[count] + temp_angle
            if scale_angle[count] >= 360:
                scale_angle[count] = scale_angle[count] - 360
            tempdict.update({scale_angle[count]: idx})
            count += 1
        scaledict = tempdict
        scale_angle.sort()
        pointer_angle = pointer_angle + temp_angle
        if pointer_angle >= 360:
            pointer_angle = pointer_angle - 360
    
    return scale_angle,pointer_angle,scaledict

def get_angle(point,center):
    angle = round(math.atan2(point[1]-center[1],point[0]-center[0]) * 180 / 3.14)
    if angle >= 90:
        angle = angle - 90
    else:
        angle = angle + 270
    return angle

def judgeStr(str1):
    "判断字符串死否为数字"
    if len(str1)==0 or str1.isspace():
        return False
    
    if str1 == '.': 
        return False
    flag1 = 0
    if(str1.isdigit()): 
        return True
    for i in str1:
        if i == '.':
            flag1 += 1
            if flag1 > 1:
                return False
        elif i > '9' and i <'0':
            return False 
    return True


class Gauge(object):
    def __init__(self):
        self.center = []
        self.pointer = []
        self.scale = []
        self.gaugerange = dict()  # 表盘刻度,格式如下：
        self.gaugeread = False
    
    def gaugeRead(self):
        scaledict = dict()
        if self.center and self.pointer and self.scale:  # sf6 直接通过中心点与指针点和刻度点的角度计算读数
            nums_angle = []
            for i,num in enumerate(self.scale):
                angle = get_angle(num,self.center)
                nums_angle.append(angle)
                scaledict.update({angle:i})
            nums_angle.sort()
            pointer_angle = get_angle(self.pointer,self.center)
            nums_angle,pointer_angle,scaledict = get_startpoint(nums_angle,pointer_angle,scaledict)
            if pointer_angle < nums_angle[0]:
                if nums_angle[0] - pointer_angle <= 3:
                    pointer_angle = nums_angle[0]
                else:
                    return False
            if pointer_angle > nums_angle[-1]:
                if pointer_angle - nums_angle[-1] <= 3:
                    pointer_angle = nums_angle[-1]
                else:
                    return False
            # if pointer_angle<nums_angle[0] or pointer_angle>nums_angle[-1]:
            # return False

            ids = len(nums_angle) - 1
            for i in list(range(ids)):
                if pointer_angle >= nums_angle[i] and pointer_angle <= nums_angle[i + 1]:
                    temp_idx_before = i
                    temp_idx_next = i+1
                    point_idx_before = scaledict[nums_angle[i]]
                    point_idx_next = scaledict[nums_angle[i+1]]
                    s_before = encoding(self.scale[point_idx_before])
                    s_next = encoding(self.scale[point_idx_next])
                    while temp_idx_before>=0 and not s_before in self.gaugerange.keys():
                        temp_idx_before -= 1
                        if temp_idx_before < 0:
                            break
                        s_before = encoding(self.scale[scaledict[nums_angle[temp_idx_before]]])
                    while temp_idx_next<=ids and not s_next in self.gaugerange.keys():
                        temp_idx_next += 1
                        if temp_idx_next > ids:
                            break
                        s_next = encoding(self.scale[scaledict[nums_angle[temp_idx_next]]])
                    if temp_idx_before >= 0 and temp_idx_next <= ids:
                        if not (judgeStr(self.gaugerange[s_before]) and judgeStr(self.gaugerange[s_next])):
                            return False
                        rate = (pointer_angle - nums_angle[temp_idx_before]) / (nums_angle[temp_idx_next]-nums_angle[temp_idx_before])
                        self.gaugeread = float(self.gaugerange[s_before]) + rate*(float(self.gaugerange[s_next]) - float(self.gaugerange[s_before]))
                        return True
                    else:
                        self.gaugeread = (pointer_angle-nums_angle[0]) / (nums_angle[ids]-nums_angle[0])
                        return True
        elif self.pointer and self.scale:   # blq ，先拟合圆再读数
            if(len(self.scale)<=3):
                return False
            ipoint = np.array(self.scale)
            circle = leastsq(residuals, [0, 0, 1], ipoint)
            x, y, r = circle[0]
            center = [x,y]
            self.center = center
            nums_angle = []
            pointer_angle = get_angle(self.pointer,center)
            for i in range(len(self.scale)):
                angle = get_angle(self.scale[i],center)
                nums_angle.append(angle)
                scaledict.update({angle:i})
            nums_angle.sort()
            # 开始读数
            if pointer_angle < nums_angle[0]:
                if nums_angle[0] - pointer_angle < 5:
                    pointer_angle = nums_angle[0]
                else :
                    return False
            if pointer_angle > nums_angle[-1]:
                if pointer_angle - nums_angle[-1] < 5:
                    pointer_angle = nums_angle[-1]
                else:
                    return False
            for i in range(len(nums_angle)-1):
                if pointer_angle >= nums_angle[i] and pointer_angle <= nums_angle[i+1]:
                    temp_idx_before = i
                    temp_idx_next = i+1
                    point_idx_before = scaledict[nums_angle[i]]
                    point_idx_next = scaledict[nums_angle[i+1]]
                    s_before = encoding(self.scale[point_idx_before])
                    s_next = encoding(self.scale[point_idx_next])
                    while temp_idx_before>=0 and not s_before in self.gaugerange.keys():
                        temp_idx_before -= 1
                        if temp_idx_before < 0 :
                            break
                        s_before = encoding(self.scale[scaledict[nums_angle[temp_idx_before]]])
                    while temp_idx_next<=(len(nums_angle)-1) and not s_next in self.gaugerange.keys():
                        temp_idx_next += 1
                        if temp_idx_next > (len(nums_angle) - 1):
                            break
                        s_next = encoding(self.scale[scaledict[nums_angle[temp_idx_next]]])
                    if temp_idx_before>=0 and temp_idx_next<=(len(nums_angle)-1):
                        rate = (pointer_angle - nums_angle[temp_idx_before]) / (nums_angle[temp_idx_next]-nums_angle[temp_idx_before])
                        if not (judgeStr(self.gaugerange[s_before]) and judgeStr(self.gaugerange[s_next])):
                            return False
                        self.gaugeread = float(self.gaugerange[s_before]) + rate*(float(self.gaugerange[s_next]) - float(self.gaugerange[s_before]))
                        return True
                    else:
                        self.gaugeread = (pointer_angle-nums_angle[0]) / (nums_angle[len(nums_angle)-1]-nums_angle[0])
                        return True
        else:
            return False

    def draw_image(self,image):
        draw = ImageDraw.Draw(image)
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 60)
        if self.gaugeread and self.center:
            print("gauge center:{}  gauge read:{}".format(self.center,round(self.gaugeread,2)))
            draw.text((round(self.center[0]),round(self.center[1])),str(round(self.gaugeread,2)),font=fnt,fill=(255,0,0,128))
        del draw
        return image

def readJsonGetGauge(jsonfile):
    with open(jsonfile,'r') as f:
        json_data = json.load(f)
    center = []
    pointer = []
    scale = []
    num_p = []
    num_data = []

    for shape in json_data["shapes"]:
        if shape["label"] == "center":
            center.append(shape["points"][0])
        elif shape["label"] == "index":
            pointer.append(shape["points"][0])
        elif shape["label"] == "num":
            scale.append(shape["points"][0])
        else:
            if shape["shape_type"] == "rectangle" and not shape["label"].startswith("f_"):
                point_n = get_center(shape["points"][0],shape["points"][1])
                num_p.append(point_n)
                num_data.append(shape["label"])
    
    # 划分表盘
    if center:
        # 存在中心说明是sf6图片，可以有多个表盘
        gauge = get_gauge(center,pointer,scale)
        #print(gauge[0].center)
        #print(gauge[0].pointer)
        #print(gauge[0].scale)
    else:
        # 没有中心点说明是blq图片，默认只有一个表盘
        gauge = [Gauge()]
        if pointer:
            gauge[0].pointer = pointer[0]
        if scale:
            gauge[0].scale = scale

    rangedict = get_range(scale,num_p,num_data)
    for i in gauge:
        i.gaugerange = rangedict

    return gauge

def centernetOutGetGauge(center,pointer,scale,num_p,num_data):
    if center:
        gauge = get_gauge(center,pointer,scale)
    else:
        gauge = get_gauge_nocenter(pointer, scale)
        if not gauge:
            return False
    rangedict = get_range(scale,num_p,num_data)
    for i in gauge:
        i.gaugerange = rangedict
    return gauge


def get_gauge(centers,pointers,scales):
    gauge = [Gauge() for i in range(len(centers))]
    # 划分中心点
    for i in range(len(centers)):
        gauge[i].center = centers[i]
    # 划分指针点
    for i in range(len(pointers)):
        min_dis = 100000
        idx = -1
        for j in range(len(gauge)):
            temp = get_distence(pointers[i],gauge[j].center)
            if temp < min_dis :
                min_dis = temp
                idx = j
        if idx == -1:
            continue
        else:
            gauge[idx].pointer = pointers[i]
    # 划分刻度点
    for i in range(len(scales)):
        point = scales[i]
        min_dis = 10000000
        idx = -1
        for j in range(len(gauge)):
            if len(gauge[j].pointer)==0:
                continue
            dis1 = get_distence(point,gauge[j].center)
            dis2 = get_distence(gauge[j].center,gauge[j].pointer)
            if dis1 < 1.5*dis2 and dis1 > dis2 / 2 and dis1 < min_dis:
                min_dis = dis1
                idx = j
        gauge[idx].scale.append(scales[i])
    return gauge

def get_gauge_nocenter(pointer,scale):
    """
    默认一张图片只有一个表盘
    """
    gauges = [Gauge()]
    if len(pointer)==0:
        return False
    elif len(pointer)==2:
        p1 = pointer[0]
        p2 = pointer[1]
        if get_distence(p1,p2)<=50:
            gauges[0].pointer = [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]
            gauges[0].scale = scale
        else:
            return False
    elif len(pointer)==1:
        gauges[0].pointer = pointer[0]
        gauges[0].scale = scale
    else :
        return False

    return gauges


def get_range(scales,num_p,num_data):
    # 数字点找最近的刻度点进行匹配
    rangedict = dict()
    for i in range(len(num_p)):
        point = num_p[i]
        min_dis = 100000
        idx = -1
        for j in range(len(scales)):
            temp = get_distence(point,scales[j])
            if temp<min_dis and temp<60:
                min_dis = temp
                idx = j
        if idx==-1:
            continue
        else:
            s = encoding(scales[idx])
            tdict = {s:num_data[i]}
            rangedict.update(tdict)
        
    return rangedict

def fit_num(num_data,num_p):
    # ------------------------------------------------------------------------- #
    # 根据已得到的刻度数字以及对应关键点信息，补全未识别出来的数字点，同时对错误的数字进行修正
    # 原理：已知crnn读取的数据以及对应的刻度点，对数值进行排列将间隔最常见的作为两个刻度的结果
    # 修改错误的识别结果，补齐没有识别出的结果
    # ------------------------------------------------------------------------- #
    
    return 0


def get_center(p1,p2):
    # ------------------------------------------ #
    # 根据两点的位置获得框的中心
    # ------------------------------------------ #
    p = []
    p.append((p1[0]+p2[0])/2)
    p.append((p1[1]+p2[1])/2)
    return p

if __name__ == "__main__":
    jsonfile = "6.json"
    gauges = readJsonGetGauge(jsonfile)
    image = cv2.imread("6.jpg")
    for gauge in gauges:
        flag = gauge.gaugeRead()
        data = gauge.gaugeread
        print(flag)
        print(data)
        cv2.circle(image,tuple([int(gauge.center[0]),int(gauge.center[1])]),2,(0,0,255),-1)
        cv2.circle(image,tuple([int(gauge.pointer[0]),int(gauge.pointer[1])]),2,(0,0,255),-1)
        for point in gauge.scale:
            point_ = tuple([int(point[0]),int(point[1])])
            cv2.circle(image,point_,2,(0,0,255),-1)
            s = encoding(point)
            if not s in gauge.gaugerange.keys():
                continue
            text_ = gauge.gaugerange[encoding(point)]
            cv2.putText(image,text_,point_,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
        break


    cv2.imwrite("s.jpg",image)
    #cv2.Waitkey(0)




