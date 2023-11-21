# ---------------------------------------------- #
# 该部分代码用于查看gflops
# ---------------------------------------------- #

import torchvision
from torchstat import stat
from torchsummary import summary
from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50

if __name__ == "__main__":
    # model = CenterNet_HourglassNet({'hm': 80, 'wh': 2, 'reg':2}).train().cuda()
    # model = CenterNet_Resnet50().train().cuda()
    model = CenterNet_Resnet50()
    stat(model, (3,1088, 1952))
