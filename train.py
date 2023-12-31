# -*- coding:utf-8 -*-
#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import warnings

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50 , CenterNet_Resnet18,CenterNet_Resnet18DCN,CenterNet_Resnet50DCN  
from nets.centernet_training import get_lr_scheduler, set_optimizer_lr
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

from utils.callbacks import LossHistory
from utils.dataloader import CenternetDataset, centernet_dataset_collate
from utils.dataloader_bj import CenternetDataset_bj , centernet_dataset_collate_bj
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、训练好的权值文件保存在logs文件夹中，每10个epoch都会保存一次，如果只是训练了几个step是不会保存的，epoch和step的概念要捋清楚一下。
   在训练过程中，该代码并没有设定只保存最低损失的，因此按默认参数训练完会有100个权值，如果空间不够可以自行删除。
   这个并不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一点，为了满足大多数的需求，还是都保存可选择性高。

3、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

4、调参是一门蛮重要的学问，没有什么参数是一定好的，现有的参数是我测试过可以正常训练的参数，因此我会建议用现有的参数。
   但是参数本身并不是绝对的，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。
  
'''  

if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda            = True
    #---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关 
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    #     注意： data_bj_num_all和 data_bj_no_blqnum所用的数据集相同，但是data_bj_num_all的标签和data_bj_no_blqnum所用的标签不同
    #           其中，data_bj_num_all检测三类关键点以及刻度数字和避雷器电击数字，但是data_bj_no_blqnum不检测避雷器电击数字
    #           如果想训练自己的数据集，修改了classes_path后，后面的 train_annotation_path  和 val_annotation_path也要对应修改
    #---------------------------------------------------------------------#
    #classes_path    = 'model_data/voc_classes.txt'
    #classes_path = './data_bj_num_all/bj_classes.txt'   
    classes_path = './data_bj_no_blqnum/bj_classes.txt'   
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    #model_path      = 'model_data/centernet_resnet50_voc.pth'
    #model_path = 'pre_weights/centernet_resnet50_voc.pth'
    #model_path = 'logs/ep100-loss2.981-val_loss2.709.pth'
    #model_path  = 'logs/ep150-loss1.660-val_loss1.631.pth'
    #model_path = './logs/resnet18_768_768/ep300-loss0.686-val_loss0.659.pth'
    #model_path  = "./logs/pgmdet_512_512/ep500-loss0.935-val_loss0.891.pth"
    model_path = ""
    #------------------------------------------------------#
    #   input_shape     输入的shape大小，32的倍数
    #------------------------------------------------------#
    #input_shape     = [512, 512]
    #input_shape = [1088,1952]
    #input_shape = [512,512]
    input_shape = [768,768]
    #input_shape = [480,640]
    #-------------------------------------------#
    #   backbone        主干特征提取网络的选择
    #-------------------------------------------#

    # 原始的centernet , 但是采用不同的 backbone
    #backbone        = "resnet50"
    #backbone        = "resnet18"
    #backbone        = "resnet18dcn"
    #backbone         = "resnet50dcn"
    #backbone       = "ghostnet"
    #backbone       = "shufflenet"
    #backbone       = "mobilenetv2"
    #backbone       = "mobilenetv3"

    # 下面是四个 pgmdet 模型
    #backbone        = "pgmdet3"   # pgmdet3 : ghostnet + 不添加SCN结构的特征融合（SCN中主要包括一个SE注意力机制，具体看论文）
    #backbone        = "pgmdet4"   # pgmdet4 : ghostnet + 添加SCN结构的特征融合
    #backbone        =  "pgmdet5"  # pgmdet5 : mobilenetv2 + 不添加SCN结构的特征融合 ， 此模型后续可转化为NCNN支持的格式用于安卓程序
    backbone        =  "pgmdet6"   # pgmdet6 : mobilenetv3 + 添加SCN结构的特征融合

    # 下面两个模型是由轻量化人体骨骼关键点检测 movenet 改造而来，速度快但是精度较低
    #backbone       = "mobilenetv2_mo"  
    #backbone      = "mobilenetv3_mo"

    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #                   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #                   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #                   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #----------------------------------------------------------------------------------------------------------------------------#
    #pretrained      = False
    pretrained      = True
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
    #      
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练： 
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 4e-5。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 4e-5。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从主干网络的预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 200，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 4e-5。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 200，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 4e-5。（不冻结）
    #       其中：由于从主干网络的预训练权重开始训练，主干的权值不一定适合目标检测，需要更多的训练跳出局部最优解。
    #             UnFreeze_Epoch可以在200-300之间调整，YOLOV5和YOLOX均推荐使用300。
    #             Adam相较于SGD收敛的快一些。因此UnFreeze_Epoch理论上可以小一点，但依然推荐更多的Epoch。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 150
    Freeze_batch_size   = 32
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 500
    Unfreeze_batch_size = 8
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    #------------------------------------------------------------------#
    #Freeze_Train        = True
    Freeze_Train        = False
    
    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    #Init_lr             = 5e-4
    Init_lr             = 5e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #------------------------------------------------------------------#
    num_workers         = 4

    #------------------------------------------------------#
    #   train_annotation_path   训练图片路径和标签
    #   val_annotation_path     验证图片路径和标签
    #------------------------------------------------------#
    #train_annotation_path   = '2007_train.txt'
    #val_annotation_path     = '2007_val.txt'
    #train_annotation_path   = './data_bj_num_all/train.txt'
    #val_annotation_path     = './data_bj_num_all/test.txt'
    train_annotation_path   = './data_bj_no_blqnum/train.txt'
    val_annotation_path     = './data_bj_no_blqnum/test.txt'
    
    #----------------------------------------------------#
    #   获取classes
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    if backbone == "resnet50":
        model = CenterNet_Resnet50(num_classes, pretrained = pretrained)
    elif backbone == "resnet50dcn":
        model = CenterNet_Resnet50DCN(num_classes,pretrained = pretrained)
    elif backbone == "resnet18": 
        model = CenterNet_Resnet18(num_classes, pretrained = pretrained)
    elif backbone == "resnet18dcn":
        model = CenterNet_Resnet18DCN(num_classes,pretrained = pretrained)
    elif backbone == "mobilenetv2_mo":
        model = MoveNet_v2(num_classes,mode="train")
    elif backbone == "mobilenetv3_mo":
        model = MoveNet_v3(num_classes,mode="train")
    elif backbone == "ghostnet":
        model = ghost_net(num_classes=num_classes)
    elif backbone == "shufflenet":
        model = ShuffleNetV2(num_classes=num_classes)
    elif backbone == "mobilenetv2":
        model = mobilenetv2(num_classes=num_classes)
    elif backbone == "mobilenetv3":
        model = mobilenetv3(n_class=num_classes,mode="large")
    elif backbone == "pgmdet3":
        model = PGMDet_3(num_classes=num_classes)
    elif backbone == "pgmdet4":
        model = PGMDet_4(num_classes=num_classes)
    elif backbone == "pgmdet5":
        model = PGMDet_5(num_classes=num_classes)
    elif backbone == "pgmdet6":
        model = PGMDet_6(num_classes=num_classes)
    else:
        raise "No Model!"

    if model_path != '':
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    loss_history    = LossHistory(save_dir, model, input_shape=input_shape)

    save_dir        = os.path.join(save_dir, "loss_" + str(loss_history.time_str))
    
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()
                        
        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 2.5e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
        #train_dataset   = CenternetDataset(train_lines, input_shape, num_classes, train = True)
        #val_dataset     = CenternetDataset(val_lines, input_shape, num_classes, train = False)
        #gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
        #                            drop_last=True, collate_fn=centernet_dataset_collate)
        #gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
        #                            drop_last=True, collate_fn=centernet_dataset_collate)

        train_dataset   = CenternetDataset_bj(train_lines, input_shape, num_classes, train = True)
        val_dataset     = CenternetDataset_bj(val_lines, input_shape, num_classes, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=centernet_dataset_collate_bj)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=centernet_dataset_collate_bj)
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 2.5e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                model.unfreeze_backbone()

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen     = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=centernet_dataset_collate)
                gen_val = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=centernet_dataset_collate)

                UnFreeze_flag = True
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, backbone, save_period, save_dir)
            
        loss_history.writer.close()
