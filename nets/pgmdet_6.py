"""
@Fire
https://github.com/fire717
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

"""
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
        padding=0, dilation=1, groups=1, bias=True))
"""

def conv_3x3_act(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_act(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_act2(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )



def dw_conv(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup)
    )

def dw_conv2(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

def dw_conv3(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )



def upsample(inp, oup, scale=2):
    return nn.Sequential(
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp),
                nn.ReLU(inplace=True),
                conv_1x1_act2(inp,oup),
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))


def IRBlock(oup, hidden_dim):
    return nn.Sequential(
            # pw
            nn.Conv2d(oup, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=False),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=False),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, n):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.n = n


        self.conv1 = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

        self.conv2 = torch.nn.ModuleList()
        for i in range(n):
            self.conv2.append(IRBlock(oup, hidden_dim))

    def forward(self, x):
        x = self.conv1(x)

        for i in range(self.n):
            x = x + self.conv2[i](x)

        return x


class Head(nn.Module):
    def __init__(self,inp,hidden_channel=96,num_classes=80,bn_momentum=0.1):
        super(Head,self).__init__()

        self.cls_head = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, hidden_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, num_classes, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.wh_head = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, hidden_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, 2, 1, 1, 0, bias=False),
        )

        self.reg_head = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, hidden_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, 2, 1, 1, 0, bias=False),
        )

    def forward(self,x):

        hm = self.cls_head(x)
        wh = self.wh_head(x)
        offset = self.reg_head(x)

        return hm,wh,offset
    
class SCN(nn.Module):
    def __init__(self,inp,oup):
        super(SCN,self).__init__()

        self.conv1 = nn.Conv2d(inp,inp,5,1,2,bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.SE = SELayer(inp)
        self.hs = Hswish()
        self.conv2 = nn.Conv2d(inp,oup,1,1,0,bias=False)
        self.bn2 = nn.BatchNorm2d(oup)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.SE(x)
        x = self.hs(x)
        x = self.conv2(x)
        x = self.bn2(x)

        return x

    



class PGMDet_6(nn.Module):
    def __init__(self,num_classes=1000,width_mult=1.):
        super(PGMDet_6, self).__init__()

        input_channel = 32

        self.features1 = nn.Sequential(*[
                            conv_3x3_act(3, input_channel, 2),
                            dw_conv(input_channel, 16, 1),
                            InvertedResidual(16, 24, 2, 6, 1)
                        ])

        self.features2 = InvertedResidual(24, 32, 2, 6, 2)
        self.features3 = InvertedResidual(32, 64, 2, 6, 3)
        self.features4 = InvertedResidual(64, 96, 1, 6, 2)
        self.features5 = InvertedResidual(96, 160, 2, 6, 2)
        self.features6 = InvertedResidual(160, 320, 1, 6, 0)

        self.scn1 = SCN(96,96)
        self.scn2 = SCN(32,32)

        #self.conv1 = nn.Conv2d(96, 96, 3, 1, 1)
        #self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)

        self.up1 = nn.ConvTranspose2d(320,96,4,2,1)
        self.up2 = nn.ConvTranspose2d(96,32,4,2,1)
        self.up3 = nn.ConvTranspose2d(32,32,4,2,1)

        self.head = Head(32,num_classes=num_classes)

        self._initialize_weights()

    
    def forward(self,x):
        f1 = self.features1(x)
        #print(f1.shape)
        f2 = self.features2(f1)
        #print(f2.shape)
        f3 = self.features3(f2)
        #print(f3.shape)
        f4 = self.features4(f3)
        #print(f4.shape)
        f5 = self.features5(f4)
        #print(f5.shape)
        f6 = self.features6(f5)
        #print(f6.shape)


        o1 = self.up1(f6)
        c1 = self.scn1(f4)

        o2 = self.up2(o1+c1)
        c2 = self.scn2(f2)

        o3 = self.up3(o2+c2)
        c3 = self.conv3(o3)

        hm,wh,offset = self.head(c3)

        return hm,wh,offset
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    from torchsummary import summary

    model = PGMDet_6(4).cuda()
    #model = MoveNet_v2().cuda()
    #print(summary(model, (3, 192, 192)))


    dummy_input1 = torch.randn(1, 3, 768, 768).cuda()
    input_names = [ "input1"] #自己命名
    output_names = [ "output1" ]
    model(dummy_input1)
    
    
    torch.onnx.export(model, dummy_input1, "mobilenetv2.onnx", 
        verbose=True, input_names=input_names, output_names=output_names,
        do_constant_folding=True,opset_version=11)
    