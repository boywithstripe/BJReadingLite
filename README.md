# CenterNet_BJ

## 1.数据集准备和模型训练

首先利用labelme对数据进行标志，标注格式见./data_bj_num_all/bj/jsons 中的相关标注文件，然后利用get_train_test.py   (目录： ./data_bj_num_all/bj/get_train_test.py)   对训练集和数据集进行划分，然后利用voc_bj_annotation.py将json格式的标签转化为算法所支持的格式。当前数据集已经转化好了，检测3类关键点和2类数字(刻度数字和避雷器电击数字)的训练数据和标签放在 ./data_bj_num_all下 ， 检测3类关键点以及1类数字(仅刻度数字)的训练集和标签放在 ./data_bj_no_blqnum下。



## 2.模型训练

修改train.py中的相关参数即可对模型进行训练，训练相关参数可以自行调整，需要注意的是，当训练不同模型时直接修改backbone即可，下面是本项目所支持的集中模型，输入尺寸默认选择[768,768]即可。

```python
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
```



当训练自己的数据集时，需要修改：

1. classes_path : 类别文件

```python
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
```

2. 训练集和测试集图片和标签目录

```python
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
```

然后运行：

```python
python train.py
```

即可进行训练，训练过程中的相关内容保存至 ./logs下。



## 3.模型测试

### 3.1关键点检测结果测试

​	使用centernet_bj.py和 get_map_bj.py可对关键点进行检测结果进行测试

​	首选需要修改centernet_bj.py中的相关参数：

​	1.所测模型路径( "model_path") 

```python
        #"model_path"        : './logs/resnet18_768_768/ep300-loss0.686-val_loss0.659.pth',  # 对应的backbone为 resnet18 , input_shape为 [768,768]
        #"model_path"         : "./logs/mobilenetv3_768_768/ep500-loss0.745-val_loss0.711.pth", # 对应的backbone为 mobilenetv3 , input_shape为 [768,768]
        #"model_path"         : "./logs/mobilenetv2_768_768/ep500-loss0.785-val_loss0.738.pth", # 对应的backbone为 mobilenetv2 , input_shape为 [768,768]
        "model_path"          : "./logs/resnet50_768_768/ep300-loss0.639-val_loss0.623.pth",   # 对应的backbone为 resnet50 , input_shape为 [768,768]
        #"model_path"          : "./logs/shufflenetv2_768_768/ep500-loss0.836-val_loss0.801.pth", # 对应的backbone为shufflenetv2, inout_shape为 [768,768]
        #"model_path"          : "./logs/ghostnet_768_768/ep500-loss0.798-val_loss0.731.pth", # 对应的backbone为ghostnet, input_shape为 [768,768]
        #"model_path"           :"./logs/mobilenetv2_mo_768_768/ep300-loss0.944-val_loss0.849.pth", # 对应的backbnoe为mobilenetv2_mo , input_shape为[768,768]
        #"model_path"          : "./logs/pgmdet_ghost_768_768/ep500-loss0.751-val_loss0.677.pth", # 对应的backbone为 pgmdet3 , input_shape为[768，768]
        #"model_path"          : "./logs/pgmdet_ghost_scn_768_768/ep500-loss0.746-val_loss0.696.pth", # 对应的backbone为 pgmdet4 , input_shape为[768,768]
        #"model_path"          : "./logs/pgmdet_mobilenetv2_768_768/ep500-loss0.804-val_loss0.714.pth", # 对应的backbone为 pgmdet5 , input_shape为[768,768]
        #"model_path"          : "./logs/pgmdet_mobilenetv2_scn_768_768/ep500-loss0.778-val_loss0.692.pth", # 对应的backbone为 pgmdet6 , input_shape为[768,768]
```

2. 类别文件目录，此部分会影响模型的导入，类别文件出错会导致模型导入出错。

```python
        #"classes_path"       : "data_bj_no_blqnum/bj_classes.txt",
        "classes_path"      : 'data_bj_num_all/bj_classes.txt',
    	
        注意：data_bj_no_blqnum 和 data_bj_num_all 数据集相同但是标签略有差别，上述的一些模型采用的标签略有不同，其中resnet18、resnet50、mobilenetv2、mobilenetv3、ghostnet、shufflenetv2均为 data_bj_num_all/bj_classes.txt
        pgmdet3、pgmdet4、pgmdet5、pgmdet6为data_bj_no_blqnum/bj_classes.txt
        
```

3.将get_map_bj.py中map_mode修改为5 ,  map_mode=5 为 测试关键点检测结果

​	修改get_map_bj.py中的classes_path，VOCdevkit_path 、test_json使其与2中的classes_path进行对应，最后

运行:

```
python get_map_bj.py
```

可对关键点进行测试。

运行结果如下（示例，模型为resnet18的结果）：

![image-20230227211929064](C:\Users\guo\AppData\Roaming\Typora\typora-user-images\image-20230227211929064.png)

### 3.2刻度数字检测结果测试

步骤与3.1类似，不过此时选用 map_mode = 0 , 

运行：

```
python get_map_bj.py
```

运行结果如下（示例，模型为resnet18的结果）：

![image-20230227212321133](C:\Users\guo\AppData\Roaming\Typora\typora-user-images\image-20230227212321133.png)

论文中只需要 r_num（即刻度数字的map）, r_fnum(避雷器数字)不需要

### 3.3读数结果测试

​	读数结果测试代码为 ： get_read_accuracy.py



### 3.4输入图片测试

