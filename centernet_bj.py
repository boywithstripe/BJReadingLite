# 关于表计的检测模型，即：既检测表计的关键点信息又检测目标框信息
# 只做了检测的工作，不涉及读数

import colorsys
import os
import time
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import ImageDraw, ImageFont
from PIL import Image

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50 , CenterNet_Resnet18,CenterNet_Resnet18DCN,CenterNet_Resnet50DCN
from nets.centernet_mobilenetv2 import MoveNet_v2
from nets.centernet_mobilenetv3 import MoveNet_v3
from nets.ghostnet import ghost_net
from nets.shufflenetv2 import ShuffleNetV2
from nets.mobilenetv2 import mobilenetv2
from nets.mobilenetv3 import mobilenetv3
from nets.pgmdet_3 import PGMDet_3
from nets.pgmdet_4 import PGMDet_4
from nets.pgmdet_5 import PGMDet_5
from nets.pgmdet_6 import PGMDet_6

from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_bbox, postprocess ,get_point , get_point_bj , get_point_box_bj

bj_classes = ["center","num","index","r_num","r_fnum"]
#bj_classes = ["center","num","index","r_num"]
point_out_path = "/media/deeppc/data/work_space/guo_workspace/centernet-pytorch-main/map_out/"

# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和backbone
#   都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
# --------------------------------------------#
class CenterNet(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        #"model_path"        : './logs/resnet18_768_768/ep300-loss0.686-val_loss0.659.pth',  # 对应的backbone为 resnet18 , input_shape为 [768,768]
        #"model_path"         : "./logs/mobilenetv3_768_768/ep500-loss0.745-val_loss0.711.pth", # 对应的backbone为 mobilenetv3 , input_shape为 [768,768]
        #"model_path"         : "./logs/mobilenetv2_768_768/ep500-loss0.785-val_loss0.738.pth", # 对应的backbone为 mobilenetv2 , input_shape为 [768,768]
        #"model_path"          : "./logs/resnet50_768_768/ep300-loss0.639-val_loss0.623.pth",   # 对应的backbone为 resnet50 , input_shape为 [768,768]
        #"model_path"          : "./logs/shufflenetv2_768_768/ep500-loss0.836-val_loss0.801.pth", # 对应的backbone为shufflenetv2, inout_shape为 [768,768]
        #"model_path"          : "./logs/ghostnet_768_768/ep500-loss0.798-val_loss0.731.pth", # 对应的backbone为ghostnet, input_shape为 [768,768]
        #"model_path"           :"./logs/mobilenetv2_mo_768_768/ep300-loss0.944-val_loss0.849.pth", # 对应的backbnoe为mobilenetv2_mo , input_shape为[768,768]
        #"model_path"          : "./logs/pgmdet_ghost_768_768/ep500-loss0.751-val_loss0.677.pth", # 对应的backbone为 pgmdet3 , input_shape为[768，768]
        #"model_path"          : "./logs/pgmdet_ghost_scn_768_768/ep500-loss0.746-val_loss0.696.pth", # 对应的backbone为 pgmdet4 , input_shape为[768,768]
        "model_path"          : "./logs/pgmdet_mobilenetv2_768_768/ep500-loss0.804-val_loss0.714.pth", # 对应的backbone为 pgmdet5 , input_shape为[768,768]
        #"model_path"          : "./logs/pgmdet_mobilenetv2_scn_768_768/ep500-loss0.778-val_loss0.692.pth", # 对应的backbone为 pgmdet6 , input_shape为[768,768]

        "classes_path"       : "data_bj_no_blqnum/bj_classes.txt",
        #"classes_path"      : 'data_bj_num_all/bj_classes.txt',
        # --------------------------------------------------------------------------#
        #   用于选择所使用的模型的主干
        #   resnet50, hourglass , resnet18, resnet50dcn, resnet18dcn , mobilenetv2_mo , mobilenetv3_mo
        #   mobilenetv2,mobilenetv3,shufflenet,ghostnet,ghostnet_new,dbnet
        # --------------------------------------------------------------------------#
        #"backbone": "resnet50",
        #"backbone": "resnet18",
        #"backbone": 'mobilenetv2',
        #"backbone": "ghostnet",
        #"backbone":"mobilenetv3",
        #"backbone": "shufflenet",
        #"backbone":"mobilenetv2_mo",
        #"backbone"  :  "pgmdet3",
        #"backbone" : "pgmdet4",
        "backbone" : "pgmdet5",
        #"backbone" : "pgmdet6",
        # --------------------------------------------------------------------------#
        #   输入图片的大小，设置成32的倍数
        # --------------------------------------------------------------------------#
        #"input_shape"       : [1088 , 1952],
        #"input_shape"       : [512, 512],
        "input_shape"       : [768, 768],
        #"input_shape": [768, 768],
        #"input_shape":[480,640],
        # --------------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # --------------------------------------------------------------------------#
        "confidence": 0.3,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # --------------------------------------------------------------------------#
        #   是否进行非极大抑制，可以根据检测效果自行选择
        #   backbone为resnet50时建议设置为True、backbone为hourglass时建议设置为False
        # --------------------------------------------------------------------------#
        "nms": True,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": False,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        # "cuda": True,
        "cuda": True,

        # ----------------------------------#
        # 是否绘制中心点(用于关键点检测)
        # ----------------------------------#
        "get_points": True,

        # ----------------------------------#
        # 是否转化为.onnx模型
        # ----------------------------------#
        "export_model": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化centernet
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   计算总的类的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        assert self.backbone in ['resnet50', 'hourglass', 'resnet18','resnet18dcn','resnet50dcn','mobilenetv2_mo','mobilenetv3_mo',
                                    "mobilenetv2","mobilenetv3","shufflenet","ghostnet","pgmdet3","pgmdet4","pgmdet5","pgmdet6"]
        if self.backbone == "resnet50":
            self.net = CenterNet_Resnet50(num_classes=self.num_classes, pretrained=False)
        elif self.backbone == "resnet50dcn":
            self.net = CenterNet_Resnet50DCN(num_classes=self.num_classes, pretrained=False)
        elif self.backbone == "resnet18":
            self.net = CenterNet_Resnet18(num_classes=self.num_classes, pretrained=False)
        elif self.backbone == "resnet18dcn":
            self.net = CenterNet_Resnet18DCN(num_classes=self.num_classes, pretrained=False)
        elif self.backbone == "mobilenetv2_mo":
            self.net = MoveNet_v2(self.num_classes,mode="train")
        elif self.backbone == "mobilenetv3_mo":
            self.net = MoveNet_v3(self.num_classes,mode="train")
        elif self.backbone == "ghostnet":
            self.net = ghost_net(num_classes=self.num_classes)
        elif self.backbone == "shufflenet":
            self.net = ShuffleNetV2(num_classes=self.num_classes)
        elif self.backbone == "mobilenetv2":
            self.net = mobilenetv2(num_classes=self.num_classes)
        elif self.backbone == "mobilenetv3":
            self.net = mobilenetv3(n_class=self.num_classes,mode="large")
        elif self.backbone == "pgmdet3":
            self.net = PGMDet_3(num_classes=self.num_classes)
        elif self.backbone == "pgmdet4":
            self.net = PGMDet_4(num_classes=self.num_classes)
        elif self.backbone == "pgmdet5":
            self.net = PGMDet_5(num_classes=self.num_classes)
        elif self.backbone == "pgmdet6":
            self.net = PGMDet_6(num_classes=self.num_classes)
        else:
            self.net = CenterNet_HourglassNet({'hm': self.num_classes, 'wh': 2, 'reg': 2})

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        if self.export_model:
            model = self.net
            model.to("cpu")
            dummy_input = torch.randn(1, 3, self.input_shape[0], self.input_shape[1], device="cpu")
            print("dummy_input.shape:{}".format(dummy_input.shape))
            torch.onnx.export(self.net, dummy_input, self.model_path.split('.pth')[0] + '.onnx', opset_version=11,
                              verbose=False, output_names=["hm", "msak", "reg"])

        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, crop=False):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            # -----------------------------------------------------------#
            #   利用预测结果进行解码
            # -----------------------------------------------------------#
            point , box = get_point_box_bj(outputs[0], bj_classes ,outputs[1], outputs[2], self.confidence, self.cuda)
            #self.get_point_txt(point,image,bj_classes,point_out_path)
            point[0][:,0] = (point[0][:,0]*4 / self.input_shape[1] * image_shape[1])
            point[0][:,1] = (point[0][:,1]*4 / self.input_shape[0] * image_shape[0])

        return point,box


    def get_FPS(self, image, test_interval):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            # -----------------------------------------------------------#
            #   利用预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

            # -------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            # -------------------------------------------------------#
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                # ---------------------------------------------------------#
                outputs = self.net(images)
                if self.backbone == 'hourglass':
                    outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
                # -----------------------------------------------------------#
                #   利用预测结果进行解码
                # -----------------------------------------------------------#
                outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

                # -------------------------------------------------------#
                #   对于centernet网络来讲，确立中心非常重要。
                #   对于大目标而言，会存在许多的局部信息。
                #   此时对于同一个大目标，中心点比较难以确定。
                #   使用最大池化的非极大抑制方法无法去除局部框
                #   所以我还是写了另外一段对框进行非极大抑制的代码
                #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
                # -------------------------------------------------------#
                results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image,
                                      self.nms_iou)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time


    # 保留满足条件的关键点信息，只保留关键点，不保留其他信息
    def get_point_txt(self,image_id,image,bj_classes,point_out_path):
        f = open(os.path.join(point_out_path, "detection-points/" + image_id + ".txt"), "w")
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            # -----------------------------------------------------------#
            #   利用预测结果进行解码
            # -----------------------------------------------------------#
            #outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)
            point , box = get_point_box_bj(outputs[0], bj_classes ,outputs[1], outputs[2], self.confidence, self.cuda)
            #print(image_id + ".jpg")
            #print(point)
            for point_ in point[0].cpu().numpy():
                class_name = bj_classes[int(point_[2])]
                x = int(point_[0]*4 / self.input_shape[1] * image_shape[1])
                y = int(point_[1]*4 / self.input_shape[0] * image_shape[0])
                f.write("{} {} {}\n".format(class_name,x,y))

        f.close()

    
    # 保留所有有关信息，包括关键点和目标区域
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            # -----------------------------------------------------------#
            #   利用预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

            # -------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            # -------------------------------------------------------#
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

            # --------------------------------------#
            #   如果没有检测到物体，则返回原图
            # --------------------------------------#
            if results[0] is None:
                return

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return

if __name__ == "__main__":
    test_txt_file = "/media/deeppc/data/work_space/guo_workspace/centernet-pytorch-main/data_bj_num_all/test.txt"
    txt_save_path = "./map_out/detection-points/"
    image_path = "test_images/test1.jpg"   # 可以是单张图片也可以是一个文件夹
    image_save_path = "./image_save"
    test_save_path = "./test_save"
    centernet = CenterNet()
    count = 0

    # -------------------------------------------------------------------
    # mode = 0 : 获取测试集图片的关键点信息并保存为txt文件
    # mode = 1 : 对测试集图片进行推理并绘制关键点和目标框，最后保存
    # mode = 2 : 对测试集以外的单张图片或者多张图片进行推理，并保存结果
    # --------------------------------------------------------------------
    mode = 2

    # 获取测试集图片的关键点并保存为txt文件
    if mode == 0:
        if os.path.exists(txt_save_path):
            #先判断该文件夹是否已经存在， 
            if os.path.getsize(txt_save_path):
                #如果存在，判断该文件夹里面是否有东西
                for file in os.listdir(txt_save_path):
                    #删除该文件夹里面的东西
                    #先通过os.listdir获取该文件夹内的文件名列表，然后逐一删除
                    os.remove(os.path.join(txt_save_path,file)) #删除该文件
        else:  
            os.mkdir(txt_save_path)

        with open(test_txt_file , 'r') as f:
            test_lines = f.readlines()
            for line in test_lines:
                image_file = line.split(' ')[0]
                image_name = os.path.basename(image_file)
                if image_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    print(count)
                    count = count + 1
                    image = Image.open(image_file)
                    image_id = image_name.split(".")[0]
                    r_image = centernet.get_point_txt(image_id,image,bj_classes,point_out_path)
    
    # 对测试集图片进行推理并保存结果
    elif mode == 1:
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)  # 删除文件夹，用os.removedirs()不能删除不为空的文件夹
            
        os.makedirs(test_save_path)

        hsv_tuples = [(x / centernet.num_classes, 1., 1.) for x in range(centernet.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        with open(test_txt_file , 'r') as f:
            test_lines = f.readlines()
            for line in test_lines:
                image_file = line.split(' ')[0]
                image_name = os.path.basename(image_file)
                if image_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    print(count)
                    count = count + 1
                    image = Image.open(image_file)
                    image_id = image_name.split(".")[0]
                    start = time.time()
                    pointes, boxes = centernet.detect_image(image)  # pointes,boxes都为list类型,  此处list中只有一个元素，因为是单张图片输出，batch_size=1
                    end = time.time()
                    print("inference time :{}".format(end - start))
                    # 绘制图片，将关键点信息和目标框绘制
                    # 线画关键点
                    draw = ImageDraw.Draw(image)

                    print(pointes)

                    for point in pointes[0].cpu().numpy():
                        class_ind = int(point[2])
                        x = (int(point[0])-3,int(point[1])-3)
                        y = (int(point[0])+3,int(point[1])+3)
                        xy = [x,y]
                        draw.ellipse(xy,fill=colors[class_ind],outline=colors[class_ind])
                    
                    # 绘制目标框

                    image.save(os.path.join(test_save_path,image_name),quality=95)
                    del draw

    # 测试多张或者单张图片
    elif mode == 2:
        if os.path.isdir(image_path):
            for image_name in os.listdir(image_path):
                if not image_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    continue
                image = Image.open(os.path.join(image_path,image_name))
                pointes,boxes = centernet.detect_image(image)
                print(pointes)
                print(boxes)
        else:
            if image_path.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image = Image.open(image_path)
                pointes,boxes = centernet.detect_image(image)
                print(pointes)
                print(boxes)
        



        
    

    """
    img = "./data_bj_num_all/bj/imgs/(9359).jpg"
    image = Image.open(img)
    centernet = CenterNet()
    crop = False
    r_image = centernet.detect_image(image, crop=crop)
    """
    
    


