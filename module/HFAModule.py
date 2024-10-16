# writer: aqiu
# HFA module
# 2024/4/7  21:34
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
from torch import nn
import torch.nn.functional as F
BatchNorm2d = nn.BatchNorm2d
from module import GIE

model_path = "./module/stvit-base-384.pth"
device = torch.device('cuda')

# HFAM
class HFAM(nn.Module):

    def __init__(self, in_channels=32, pool_size=[32,16], norm_layer=nn.BatchNorm2d, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(HFAM, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = in_channels

        self.conv0 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))

        # STViT
        self.STVitblock = GIE.STViT(in_chans = inter_channels,
                    embed_dim=[96, 192, 384], # 52M, 9.9G, 361 FPS
                    depths=[4, 6, 14],
                    num_heads=[2, 3, 6],
                    n_iter=[1, 1, 1], 
                    stoken_size=[4, 4, 4],
                    #projection=384,                   
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.1,
                    drop_path_rate=0.3, 
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0],
                    layerscale=[False, False, True],
                    init_values=1e-6,)
        model_dict = self.STVitblock.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        temp = {}
        for k, v in pretrained_dict.items():
            try:    
                if np.shape(model_dict[k]) == np.shape(v):
                    temp[k]=v
            except:
                pass
        model_dict.update(temp)
        self.STVitblock.load_state_dict(model_dict)
        

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))

        self.conv0_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True))
        self.conv1_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 5, bias=False, dilation=5),
                                     norm_layer(inter_channels))
        self.conv1_0_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 7, bias=False, dilation=7),
                                     norm_layer(inter_channels))

        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 3, bias=False, dilation=3),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels+384, 256, 1, bias=False),
                                norm_layer(256),
                                nn.Dropout(0.1))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

        self._initialize_weights()

    def forward(self, x):
        _, _, h, w = x.size()
        # GIE
        x0 = self.STVitblock(x)
        
        #x2 = self.conv1_2(x)   # H * W * C/4
        
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x)), (h, w), **self._up_kwargs)

        # RCCA
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x0, x2], dim=1))
        
        return F.relu_(out)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
