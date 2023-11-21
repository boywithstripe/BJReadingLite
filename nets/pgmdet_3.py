import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ["pgmdet"]

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

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


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

def conv(inp,oup,kernel_size=3,stride=1,padding=0,relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )


def upsample(inp,oup,scale=2):
    return nn.Sequential(
        nn.Conv2d(inp,inp,3,1,1,groups=inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp,oup,1,1,0,bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=scale,mode="bilinear",align_corners=False)
    )

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

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

def ghost_s(input_channel,k,exp_size,c,use_se,s,width_mult):
    output_channel = _make_divisible(c * width_mult, 4)
    hidden_channel = _make_divisible(exp_size * width_mult, 4)
    #print("cfgs:inp:{},hid:{},oup:{},k:{},s:{}".format(input_channel,hidden_channel,output_channel,k,s))
    return GhostBottleneck(input_channel,hidden_channel,output_channel,k,s,use_se),output_channel


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


class PGMDet_3(nn.Module):
    def __init__(self,num_classes=1000,width_mult=1.):
        super(PGMDet_3,self).__init__()

        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv0 = conv(3,output_channel,3,2,1,True)
        input_channel = output_channel

        k = [3 ,  3,  3,  5,   5,   3,   3,  3,   3,   3,   5]    
        t = [16, 48, 72, 120, 240, 480, 480, 672, 672, 960, 960]
        c = [16, 24, 36, 60,  120,  120,  120, 160, 160, 320, 320]
        SE= [ 0,  0,  0,  1,   1,   0,   0, 0,   1,   1,   1]
        s = [ 1,  2,  1,  2,   1,   2,   1, 1,   1,   2,   1]

        self.ghost0 ,input_channel = ghost_s(input_channel,k[0],t[0],c[0],SE[0],s[0],width_mult)

        self.ghost1, input_channel = ghost_s(input_channel, k[1], t[1], c[1], SE[1], s[1], width_mult)

        self.ghost2, input_channel = ghost_s(input_channel, k[2], t[2], c[2], SE[2], s[2], width_mult)
    

        self.ghost3, input_channel = ghost_s(input_channel, k[3], t[3], c[3], SE[3], s[3], width_mult)

        self.ghost4, input_channel = ghost_s(input_channel, k[4], t[4], c[4], SE[4], s[4], width_mult)
        scn_inp1 = input_channel

        self.ghost5, input_channel = ghost_s(input_channel, k[5], t[5], c[5], SE[5], s[5], width_mult)

        self.ghost6, input_channel = ghost_s(input_channel, k[6], t[6], c[6], SE[6], s[6], width_mult)

        self.ghost7, input_channel = ghost_s(input_channel, k[7], t[7], c[7], SE[7], s[7], width_mult)
        scn_inp2 = input_channel

        self.ghost8, input_channel = ghost_s(input_channel, k[8], t[8], c[8], SE[8], s[8], width_mult)

        self.ghost9, input_channel = ghost_s(input_channel, k[9], t[9], c[9], SE[9], s[9], width_mult)

        self.ghost10, input_channel = ghost_s(input_channel, k[10], t[10], c[10], SE[10], s[10], width_mult)
        scn_inp3 = input_channel

        #self.ghost11, input_channel = ghost_s(input_channel, k[11], t[11], c[11], SE[11], s[11], width_mult)

       # self.ghost12, input_channel = ghost_s(input_channel, k[12], t[12], c[12], SE[12], s[12], width_mult)
        #scn_inp3 = input_channel

        #self.ghost13, input_channel = ghost_s(input_channel, k[13], t[13], c[13], SE[13], s[13], width_mult)

        #self.ghost14, input_channel = ghost_s(input_channel, k[14], t[14], c[14], SE[14], s[14], width_mult)

        #self.ghost15, input_channel = ghost_s(input_channel, k[15], t[15], c[15], SE[15], s[15], width_mult)


        #self.up1 = upsample(input_channel ,input_channel,scale=2)
        #self.up2 = upsample(input_channel ,input_channel,scale=2)
        #self.up3 = upsample(input_channel, input_channel//2, scale=2)


        self.up1 = nn.ConvTranspose2d(scn_inp3,scn_inp2,4,2,1)
        self.up2 = nn.ConvTranspose2d(scn_inp2,scn_inp1,4,2,1)
        self.up3 = nn.ConvTranspose2d(scn_inp1,120,4,2,1)

        #self.scn1 = SCN(scn_inp1,input_channel)
        #self.scn2 = SCN(scn_inp2,input_channel)
        #self.scn3 = SCN(scn_inp3,input_channel)

        self.convup1 = conv(scn_inp2,scn_inp2,3,1,1)
        self.convup2 = conv(scn_inp1,scn_inp1,3,1,1)
        self.convup3 = conv(scn_inp1,scn_inp1,3,1,1)

        self.head = Head(scn_inp1,num_classes=num_classes)

        self._initialize_weights()

    def forward(self,x):
        x = self.conv0(x)    # k=3x3 , s=2

        #print("g1:{}".format(x.shape))
        x = self.ghost0(x)   #
        print("g0:{}".format(x.shape))
        x = self.ghost1(x)
        print("g1:{}".format(x.shape))
        g2 = self.ghost2(x)
        print("g2:{}".format(g2.shape))
        x = self.ghost3(g2)
        print("g3:{}".format(x.shape))
        g4 = self.ghost4(x)
        print("g4:{}".format(g4.shape))
        x = self.ghost5(g4)
        print("g5:{}".format(x.shape))
        x = self.ghost6(x)
        print("g6:{}".format(x.shape))
        g7 = self.ghost7(x)
        print("g7:{}".format(g7.shape))
        x = self.ghost8(g7)
        print("g8:{}".format(x.shape))
        x = self.ghost9(x)
        print("g9:{}".format(x.shape))
        g10 = self.ghost10(x)
        print("g10:{}".format(g10.shape))

        o1 = self.up1(g10)
        y1 = self.convup1(g7)

        o2 = self.up2(o1+y1)
        y2 = self.convup2(g4)

        o3 = self.up3(o2+y2)
        y3 = self.convup3(o3)

        hm, wh, offset = self.head(y3)

        return hm, wh, offset

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__=='__main__':
    from torchsummary import summary
    import time

    model = PGMDet_3(num_classes=4).cuda()  # 4种热力图输出

    dummy_input1 = torch.randn(1, 3, 768, 768).cuda()


    start = time.time()
    model(dummy_input1)
    end = time.time()

    print("inference time:{}".format(end-start))

    input_names = ["input1"]  # 自己命名
    output_names = ["output1"]

    """
    torch.onnx.export(model, dummy_input1, "pgmdet_ghost.onnx",
                      verbose=True, input_names=input_names, output_names=output_names,
                      do_constant_folding=True, opset_version=11)
    """