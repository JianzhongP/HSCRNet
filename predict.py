import time

import os
import cv2
import numpy as np
from PIL import Image
'''
from pspnet_test import PSPNet
from unet_test import Unet
from deeplabv3_plus_test import DeeplabV3
from hrnet_test import HRnet_Segmentation
from CGNet_test import cg
from segformer_test import segformer
from PVT_L_test import PVT
from CVT_test import cvt
from DDRNet_test import DDRNet
from seaformer_test import seaformer
from SCTNet_test import st
'''
from HSCRNet_test import HSCRNet

'''
    UNet:         Unet()
    PSPNet:       PSPNet()
    DeeplabV3:    DeeplabV3()
    HRNet:        HRnet_Segmentation()
    CGNet:        cg()
    SegFormer:    segformer()
    PVT:          PVT()
    CVT:          cvt()
    DDRNet:       DDRNet()
    SeaFormer:    seaformer()
    SCTNet:       st()
'''

if __name__ == "__main__":
    
    model = HSCRNet()
    mode = "predict"
    count           = True
    save_path = 'img/'
    data_path = 'WHDLD/Images/' # 'landcover.ai/Images/'
    datatxt_path = 'WHDLD/segmentaion_sets/'  # 'landcover.ai/segmentaion_sets/'
    name_classes    = ['background','building', 'road', 'pavement', 'vegetation', 'bare soil', 'water']
    # name_classes    = ['background','building', 'vegetation', 'water']   # Landcover.ai
    

    if mode == "predict":
        # Overall inputs are processed.
        with open(os.path.join(datatxt_path, 'test.txt'), 'r') as test:
            line = test.readlines()
            # print(line)
            for im in line:
                im_id = im.split('\n')[0]
                img_path = os.path.join(data_path, im_id+'.jpg')
                image = Image.open(img_path)
                r_image = model.detect_image(image, im_id, count=count, name_classes=name_classes)
                # r_image.show()
                r_image.save(os.path.join(save_path, im_id+'.png'))
        '''
        # Individual inputs are processed.
        while True:
            img = input('Input image filename:')
            if img=='s':
                break
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = model.detect_image(image, img, count=count, name_classes=name_classes)
                #r_image.show()
                r_image.save(os.path.join(save_path, img.split('.')[0]+'.png'))
        '''