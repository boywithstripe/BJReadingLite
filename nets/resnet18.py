from __future__ import absolute_import, division, print_function

import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=1000):
        self.inplanes = 64
        super(ResNet,self).__init__()   # super().__init() 继承父类的init方法
        # 512,512,3 -> 256,256,64
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64,momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # 256,256,64 -> 128,128,64
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True)

        # 128,128,64 -> 128,128,64
        self.layer1 = self._make_layer(block,64,layers[0])

        # 128,128,64 -> 64,64,128
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)

        # 64,64,128 -> 32,32,256
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)

        # 32,32,256 -> 16,16,512
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 后面再经过三个上采样层 变成128,128,256,刚好是四倍,最后就是输出层了

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 当存在改变特征层宽高或者改变特征层通道时加一个存在卷积的桥梁
            downsample = nn.Sequential(  # 以相关的顺序添加到容器中去
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),  # 添加BatchNorm2d进行数据的归一化处理
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):  # 不改变输出特征层的宽高和通道数
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(pretrained = True):
    model = ResNet(BasicBlock,[2,2,2,2])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18'],model_dir='model_data/')
        model.load_state_dict(state_dict)
    # -------------------------------------------------------------#
    # 获取特征提取部分
    # -------------------------------------------------------------#

    features = list([model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.layer4])
    features = nn.Sequential(*features)
    return features


class resnet18_Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(resnet18_Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False

        # ----------------------------------------------------------#
        #   16,16,512 -> 32,32,256 -> 64,64,256 -> 128,128,256
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        # ----------------------------------------------------------#
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 256, 256],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class resnet18_Head(nn.Module):
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1):
        super(resnet18_Head, self).__init__()
        # -----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 256 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        # -----------------------------------------------------------------#
        # 热力图预测部分
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(256, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))

        # 中心点偏移量预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(256, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()  
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        return hm, wh, offset
    
if __name__=="__main__":

    from torchsummary import summary
    model = resnet18(pretrained = True).cuda()
    print(summary(model,(3,768,768)))


    dummy_input1 = torch.randn(1,3,768,768).cuda()
    input_names = [ "input1"] #自己命名
    output_names = [ "output1" ]
    torch.onnx.export(model, dummy_input1, "shufflenetv2.onnx", 
        verbose=True, input_names=input_names, output_names=output_names,
        do_constant_folding=True,opset_version=11)

    from thop import profile
    input = torch.randn(1,3,768,768).cuda()
    flops,params = profile(model,inputs=(input,))
    print('the flops is {}G,the params is {}M'.format(round(flops/(10**9),2), round(params/(10**6),2)))
    

