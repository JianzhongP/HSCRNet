import os

from PIL import Image
from tqdm import tqdm

from HSCRNet_test import HSCRNet
'''
from unet_test import Unet
from pspnet_test import PSPNet
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
from utils.utils_metrics import compute_mIoU, show_results
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
    unet:         Unet()
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
    #---------------------------------------------------------------------------#
    #   miou_mode = 0: represents the entire miou computation process, including obtaining prediction results, and computing miou.
    #   miou_mode = 1: represents merely obtaining predicted results.
    #   miou_mode = 2: Represents a mere calculation of miou.
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    num_classes     = 7
    name_classes    = ['background','bare soil', 'building', 'pavement', 'vegetation', 'road', 'water']
    # name_classes    = ["background", "building", "vegetation", "water"] #Landcover.ai
    WHDLD_path  = 'WHDLD'# 'landcover.ai'

    image_ids       = open(os.path.join(WHDLD_path, "segmentaion_sets/test.txt"),'r').read().splitlines()
    gt_dir          = os.path.join(WHDLD_path, "Labels/")
    miou_out_path   = "evalution_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        model = HSCRNet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(WHDLD_path, "Images/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = model.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print(IoUs.shape)
        print("Get miou done.")
        name_classes = ['building', 'road', 'pavement', 'vegetation', 'bare soil', 'water']
        # name_classes    = ["building", "vegetation", "water"]
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)