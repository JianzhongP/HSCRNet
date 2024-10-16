import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

from module import HFAModule
from module import SCRAModule

#  backbone
class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]
        # self.down_idx = [2, 4, 9, 16]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_1 = self.features[0](x)    # 128 * 128 * 32
        low_2 = self.features[1:4](low_1)
        low_3 = self.features[4:7](low_2)
        low_4 = self.features[7:14](low_3)
        low_5 = self.features[14:](low_4)
        return low_1, low_2, low_3, low_4, low_5





#   ASPP
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
        self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=2*rate, dilation=2*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
        self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=4*rate, dilation=4*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
        self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=8*rate, dilation=8*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

    def forward(self, x):
        [b, c, row, col] = x.size()

        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x,2,True)
        global_feature = torch.mean(global_feature,3,True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class HSCR(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(HSCR, self).__init__()
        if backbone=="xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))


        self.TBCA2 = HFAModule.HFAM(in_channels=24)
        self.ELCA1 = SCRAModule.SCRAM(96)
        self.ELCA2 = SCRAModule.SCRAM(256)

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=8//downsample_factor)
        
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(96+256, 96+256, 3, 1, 1, groups=96+256),
            nn.BatchNorm2d(96+256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv2d(96+256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(256+256, 256+256, 3, 1, 1, groups=256+256),
            nn.BatchNorm2d(256+256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv2d(256+256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(256+32, 256+32, 3, 1, 1, groups=256+32),
            nn.BatchNorm2d(256+32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv2d(256+32, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        
        self.upsample1 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=2, padding=1,
                                            output_padding=1)

        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1,
                                            output_padding=1)

        self.upsample5 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1,
                                            output_padding=1)
                                            
    def forward(self, x):
        b, _, h, w = x.size()
        level_1, level_2, level_3, level_4, level_5 = self.backbone(x)

        x_level_2 = self.TBCA2(level_2)   #c = 512
        
        x_level_4 = self.ELCA1(level_4)    # 32 * 32 * 96
        final = self.aspp(level_5)         # 32 * 32 * 256
        final = self.ELCA2(final)
        
        up_1 = self.up_conv1(torch.cat((final, x_level_4), dim=1))
        up_1 = self.upsample3(up_1)
        up_2 = self.up_conv2(torch.cat((up_1, x_level_2), dim=1))
        up_2 = self.upsample5(up_2)
        up_3 = self.up_conv3(torch.cat((up_2, level_1), dim=1))
        up_3 = self.cls_conv(up_3)
        up_3 = self.upsample1(up_3)

        return up_3

