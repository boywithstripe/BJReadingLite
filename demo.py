import colorsys
import os
#from statistics import quantiles
import time
from time import sleep
import math
import sys

import numpy as np
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
from PIL import ImageDraw, ImageFont
from tqdm import tqdm

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50, CenterNet_Resnet18,CenterNet_Resnet18DCN,CenterNet_Resnet50DCN
#from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50, CenterNet_Resnet18  #,CenterNet_Resnet18DCN,CenterNet_Resnet50DCN
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_bbox, postprocess

from nets.centernet_mobilenetv2 import MoveNet_v2
from nets.centernet_mobilenetv3 import MoveNet_v3
from nets.ghostnet import ghost_net
from nets.shufflenetv2 import ShuffleNetV2
from nets.mobilenetv2 import mobilenetv2
from nets.mobilenetv3 import mobilenetv3
from nets.ghostnet_new import ghost_net_new
from nets.pgmdet import PGMDet
from nets.mobilenetv2_centernet import get_mobile_net




import crnn.utils.utils as utils
import crnn.models.crnn as crnn
import crnn.models.stn_crnn as stn_crnn
from easydict import EasyDict as edict
from torch.autograd import Variable
import yaml
import cv2
import argparse
from utils.utils_reading import Gauge,readJsonGetGauge,centernetOutGetGauge,encoding,decoding,get_distence

alphabets = "n.0123456789"


def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument("--cfg",default="crnn_file/360CC_config.yaml",help='experiment configuration filename')
    parser.add_argument('--checkpoint',default="crnn_file/crnn_model.pth",help="the path to your checkpoints")
    parser.add_argument('--stn',default=False,help="use stn module or not")
    parser.add_argument('--image_path',type=str,default="./image_huayun_test",help="test image path")
    parser.add_argument('--image_save_path',type=str,default="./test_save",help="save image path")
    
    args = parser.parse_args()

    return args

class Crnn(object):
    def __init__(self,args):
        self.model_path = args.checkpoint
        with open(args.cfg,'r') as f:
            self.config = yaml.safe_load(f)
            self.config = edict(self.config)
        self.config.DATASET.ALPHABETS = alphabets
        self.config.MODEL.NUM_CLASSES = len(self.config.DATASET.ALPHABETS)
        self.args = args

        self.generate()

    def generate(self):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        #device = torch.device('cpu')
        if not args.stn:
            self.model = crnn.get_crnn(self.config).to(device)
        else :
            self.model = stn_crnn.get_crnn(self.config).to(device)
        print('loading pretrained model from {}'.format(self.model_path))
        checkpoint = torch.load(self.model_path)
        if 'state_dict' in checkpoint.keys():
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)


    def recognition(self,img,converter,device):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h , w = img.shape

        # 保持原比例的同时对图片进行resize
        # 保持原比例的同时对 图片进行resize
        inp_h = self.config.MODEL.IMAGE_SIZE.H
        inp_w = self.config.MODEL.IMAGE_SIZE.W

        scale = min(inp_h / h, inp_w / w)
        nw = int(scale * w)
        nh = int(scale * h)

        img = cv2.resize(img, (nw, nh))
        img = np.reshape(img, (nh, nw, 1))

        image = np.zeros((inp_h, inp_w, 1))
        image[0:nh, 0:nw] = img

        # normalize
        img = image.copy()
        img = img.astype(np.float32)
        img = (img / 255. - self.config.DATASET.MEAN) / self.config.DATASET.STD
        img = img.transpose([2, 0, 1])
        img = torch.from_numpy(img)

        img = img.to(device)
        img = img.view(1, *img.size())
        self.model.eval()
        preds = self.model(img)
        #print(preds.shape)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

        #print('results: {0}'.format(sim_pred))

        return sim_pred

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
        # "model_path"        : 'logs/loss_2022_04_11_10_38_34/ep099-loss2.284-val_loss2.315.pth',
        #"model_path": './logs/loss_2022_04_27_23_12_19/ep270-loss0.674-val_loss0.660.pth',  # resnet18
        #"model_path": './logs/resnet18_768_768/ep270-loss0.674-val_loss0.660.pth',   # resnet18_768_768
        "model_path"          : "./logs/pgmdet_1_512_512/ep100-loss0.861-val_loss0.812.pth",

        "classes_path": 'data_bj_num_all/bj_classes.txt',
        # --------------------------------------------------------------------------#
        #   用于选择所使用的模型的主干
        #   resnet50, hourglass , resnet18
        # --------------------------------------------------------------------------#
        #"backbone": 'resnet18',
        #"backbone": 'resnet18dcn',
        #"backbone": 'resnet50',
        "backbone": 'pgmdet',
        #backbone  "pgmdet"
        # --------------------------------------------------------------------------#
        #   输入图片的大小，设置成32的倍数
        # --------------------------------------------------------------------------#
        "input_shape"       : [512, 512],
        # "input_shape"        : [1088,1952],
        # "input_shape"        : [512,512],
        # "input_shape": [768, 768],
        # --------------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # --------------------------------------------------------------------------#
        "confidence": 0.4,
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
        "cuda": True,

        # ----------------------------------#
        # 是否绘制中心点(用于关键点检测)
        # ----------------------------------#
        "get_points": True,

        # ----------------------------------#
        # 是否转化为.onnx模型
        # ----------------------------------#
        "export_model": False
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
        print(self.model_path)
        print(self.backbone)
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        assert self.backbone in ['resnet50', 'hourglass', 'resnet18','resnet18dcn','resnet50dcn','pgmdet']
        if self.backbone == "resnet50":
            self.net = CenterNet_Resnet50(num_classes=self.num_classes, pretrained=False)
        #elif self.backbone == "resnet50dcn":
        #    self.net = CenterNet_Resnet50DCN(num_classes=self.num_classes,pretrained=False)
        elif self.backbone == "resnet18":
            self.net = CenterNet_Resnet18(num_classes=self.num_classes, pretrained=False)
        elif self.backbone == "pgmdet":
            self.net = PGMDet(num_classes=self.num_classes)
        #elif self.backbone == "resnet18dcn":
        #    self.net = CenterNet_Resnet18DCN(num_classes=self.num_classes, pretrained=False)
        else:
            self.net = CenterNet_HourglassNet({'hm': self.num_classes, 'wh': 2, 'reg': 2})

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        if self.export_model:
            model = self.net
            model.to("cpu")
            dummy_input = torch.randn(1, 3, self.input_shape[0], self.input_shape[1], device="cpu")
            print("dummy_input.shape:{}".format(dummy_input.shape))
            torch.onnx.export(self.net, dummy_input, self.model_path.split('.pth')[0] + '.onnx', opset_version=9,
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
    def detect_image(self, image,args,crnn_model,converter,crop=False):
        center = []
        pointer = []
        scale = []
        num_p = []
        num_data = []
        image_temp = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        #crnn_model = Crnn(args)
        #converter = utils.strLabelConverter(crnn_model.config.DATASET.ALPHABETS)
        device  = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        #device  = torch.device("cpu")
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
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            point = [int(left+right)/2,int(top+bottom)/2]

            if predicted_class=="center":
                center.append(point)
            elif predicted_class=="index":
                pointer.append(point)
            elif predicted_class=="num":
                scale.append(point)

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)

            if not predicted_class.startswith("r_"):
                x = int(left + right) // 2
                y = int(top + bottom) // 2
                #print("{}  {}".format(x, y))
                xy = [(x - 5, y - 5), (x + 5, y + 5)]
                draw.ellipse(xy, fill=self.colors[c], outline=self.colors[c])
                continue

            #print(box)
            num = crnn_model.recognition(image_temp[top:bottom,left:right],converter,device)
            #print(num)
            if num.startswith('n'):
                num = '-' + num.split('n')[-1]
            if predicted_class=="r_num":
                num_p.append(point)
                num_data.append(num)
            elif predicted_class=="r_fnum":
                continue
            fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 35)
            draw.text(((left+right)/2,(top+bottom)/2), str(num), font=fnt, fill=self.colors[c])


            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            # draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image,center,pointer,scale,num_p,num_data

    def get_points(self, hm, confidence):
        # --------------------------------------------------- #
        # 根据获得的中心点
        return 0

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

def get_range(gaugedict,scale):
    # 获得表盘的量程(即：最大量程减去最小量程)
    # 由于模型有时候得到的结果不太正确，因此此处仅仅采用标签的量程
    num_data = []
    for p in scale:
        s = encoding(p)
        if s in gaugedict.keys():
            num_data.append(float(gaugedict[s]))
    num_data.sort()
    if len(num_data)<=1:
        return 1
    else :
        return num_data[-1] - num_data[0]    

if __name__ == '__main__':
    
    # 单张图片测试
    crop = False
    args = parse_arg()
    centernet = CenterNet()
    crnn_model = Crnn(args)

    converter = utils.strLabelConverter(crnn_model.config.DATASET.ALPHABETS)
    img_save_path = args.image_save_path
    img = args.image_path
    count = 0
    if os.path.isdir(img):
        img_lists = os.listdir(img)
        for name in img_lists:
            
            count += 1
            if count >= 90 and count <=220:
                continue
            print(name)
            image = Image.open(os.path.join(img,name))
            t1 = time.time()
            image,center,pointer,scale,num_p,num_data = centernet.detect_image(image,args,crnn_model,converter,crop=crop)
            
            print(next(centernet.net.parameters()).device)
            print(next(crnn_model.model.parameters()).device)
            gauge_centernet = centernetOutGetGauge(center,pointer,scale,num_p,num_data)
            if not gauge_centernet:
                continue
            for i in range(len(gauge_centernet)):
                flag = gauge_centernet[i].gaugeRead()
                read_ = gauge_centernet[i].gaugeread
                image = gauge_centernet[i].draw_image(image)
            t2 = time.time()
            print("time sum : {}".format(t2-t1))
            #image.save(os.path.join(img_save_path,name),quality=95)
            #image.show()
            #sleep(4)

    elif os.path.isfile(img):
        image = Image.open(img)
        image,center,pointer,scale,num_p,num_data = centernet.detect_image(image,args,crnn_model,converter,crop=crop)
        gauge_centernet = centernetOutGetGauge(center,pointer,scale,num_p,num_data)
        if not gauge_centernet:
            raise "no gauge"
        #print("模型读数结果：")
        for i in range(len(gauge_centernet)):
            flag = gauge_centernet[i].gaugeRead()
            read_ = gauge_centernet[i].gaugeread
            image = gauge_centernet[i].draw_image(image)

        #print(img)
        #image.save("s.jpg" , quality=95)
        image.save(os.path.join(img_save_path,os.path.basename(img)),quality=95)
    
    else:
        raise "ERROR"

