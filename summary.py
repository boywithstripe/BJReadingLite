#--------------------------------------------#
#   该部分代码用于看网络参数
#--------------------------------------------#
from torchsummary import summary

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50,CenterNet_Resnet18

if __name__ == "__main__":
    # model = CenterNet_HourglassNet({'hm': 80, 'wh': 2, 'reg':2}).train().cuda()
    # summary(model,(3,128,128))
    model = CenterNet_Resnet18().train().cuda()
    #summary(model, (3, 512, 512))
    summary(model,(3,1088,1952))
