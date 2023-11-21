import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
import cv2

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
#from centernet import CenterNet
from centernet_bj import CenterNet
from utils.utils_map_point import read_json_file,get_recall_accuracy

if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，在门限值不同时，网络的Recall和Precision值是不同的。
    map计算结果中的Recall和Precision代表的是当预测时，门限置信度为0.5时，所对应的Recall和Precision值。

    此处获得的./map_out/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
    目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。
    '''
    bj_classes = ['center','num','index','r_num','r_fnum']
    # ------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map,仅对名称为 r_的计算map比如刻度数字区域。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #   map_mode为5代表仅计算三类关键点信息，并计算其对应的precision和recall
    # -------------------------------------------------------------------------------------------------------------------#
    map_mode = 5
    # -------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    # -------------------------------------------------------#
    # classes_path    = 'model_data/voc_classes.txt'
    classes_path = './data_bj_num_all/bj_classes.txt'
    # classes_path = './data_bj_archive/bj_classes.txt'
    # classes_path = './data_bj_no_blqnum/bj_classes.txt'

    # -------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    # -------------------------------------------------------#
    MINOVERLAP = 0.1
    # -------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    # -------------------------------------------------------#
    map_vis = False
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    #VOCdevkit_path = 'data_bj_num'
    VOCdevkit_path = 'data_bj_num_all/'
    #VOCdevkit_path = 'data_bj_archive/'

    # -------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    # -------------------------------------------------------#
    map_out_path = 'map_out'
    # -------------------------------------------------------#
    # 测试集json文件
    # -------------------------------------------------------#
    test_json = "./data_bj_num_all/bj/test.json"
    # test_json = "./data_bj_archive/bj/test.json"

    image_data = open(os.path.join(VOCdevkit_path, "test.txt")).read().split('\n')
    image_data.pop()
    image_path = [i.split(' ')[0] for i in image_data]   # 获得测试集的图片路径
    image_ids = [os.path.basename(i).split('.')[0] for i in image_path]  # 获得测试集的图片名称（不含有.jpg）
    # 利用字典保存所有的数据信息
    bbox_data = dict()
    for i in image_data:
        image_name = os.path.basename(i.split(' ')[0]).split('.')[0]
        box = i.split(' ')[1:]
        bbox_data.update({image_name:box})

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))
    if not os.path.exists(os.path.join(map_out_path,'detection-points')):
        os.makedirs(os.path.join(map_out_path,'detection-points'))

    class_names, _ = get_classes(classes_path)
    print(class_names)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        centernet = CenterNet(confidence=0.01, nms_iou=0.5)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "bj/imgs/" + image_id + ".jpg")
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            centernet.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2 :
        print("Get ground truth result:")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                for box in bbox_data[image_id]:
                    temp = box.split(',')
                    if len(temp) < 5:
                        continue
                    x1 = int(temp[0])
                    y1 = int(temp[1])
                    x2 = int(temp[2])
                    y2 = int(temp[3])
                    cls_id = int(temp[4])
                    new_f.write("{} {} {} {} {}\n".format(bj_classes[cls_id],x1,y1,x2,y2))
            new_f.close()
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path=map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")

    if map_mode == 5:
        if os.path.getsize(os.path.join(map_out_path,"detection-points")):
            temp_path = os.path.join(map_out_path,"detection-points") 
            for file in os.listdir(temp_path):
                os.remove(os.path.join(temp_path,file))
        print("Load model")
        centernet = CenterNet()
        print("Load model done.")
        print("get truth point result:")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "bj/imgs/" + image_id + ".jpg")
            image = Image.open(image_path)
            r_image = centernet.get_point_txt(image_id,image,bj_classes,map_out_path)
        print("get precision and recall:")
            
        data_load = read_json_file(test_json)
        get_recall_accuracy(data_load,os.path.join(map_out_path,"detection-points"))
        


