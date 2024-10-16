# -*- coding: latin-1 -*-
import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.CVT import ConvolutionalVisionTransformer
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


#-----------------------------------------------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、backbone和num_classes都需要修改！
#   如果出现shape不匹配，一定要注意训练时的model_path、backbone和num_classes的修改
#-----------------------------------------------------------------------------------#
class cvt(object):
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        #-------------------------------------------------------------------#
        "model_path"        : "logs/L_CVT_ep120-loss0.246-val_loss0.212.pth",
        #----------------------------------------#
        #   所需要区分的类的个数+1
        #----------------------------------------#
        "num_classes"       : 4,
        "input_shape"       : [256, 256],
        "mix_type"          : 1,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   初始化Deeplab
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        # 使用 _defaults 字典中的默认值初始化对象的属性
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            # 将 kwargs 字典中的每个键值对设置为对象的属性
            setattr(self, name, value)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            #self.colors = [(0, 0, 0), (255, 0, 0), (255, 255, 0), (192, 192, 0), (0, 255, 0), (128, 128, 128), (0, 0, 255)]
            self.colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 255)]
        else:
            # 创建了一个HSV颜色空间的元组列表。HSV颜色空间中的色调（H）是按照类的数量均匀分布的，
            # 饱和度（S）和值（V）都设置为最大值1。这样每个类都会有一个唯一的色调。
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            # HSV -> RGB
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            # RGB颜色通常以0到255的整数表示，将RGB颜色元组中的每个值乘以255并转换为整数，以便得到适合显示的颜色值
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   获得模型
        #---------------------------------------------------#
        self.generate()
        
        show_config(**self._defaults)
                    
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        spec = dict(INIT='trunc_norm',
    NUM_STAGES=3,
    PATCH_SIZE=[7, 3, 3],
    PATCH_STRIDE=[4, 2, 2],
    PATCH_PADDING=[2, 1, 1],
    DIM_EMBED=[64, 192, 384],
    NUM_HEADS=[1, 3, 6],
    DEPTH=[1, 2, 10],
    MLP_RATIO=[4.0, 4.0, 4.0],
    ATTN_DROP_RATE=[0.0, 0.0, 0.0],
    DROP_RATE=[0.0, 0.0, 0.0],
    DROP_PATH_RATE=[0.0, 0.0, 0.1],
    QKV_BIAS=[True, True, True],
    CLS_TOKEN=[False, False, True],
    POS_EMBED=[False, False, False],
    QKV_PROJ_METHOD=['dw_bn', 'dw_bn', 'dw_bn'],
    KERNEL_QKV=[3, 3, 3],
    PADDING_KV=[1, 1, 1],
    STRIDE_KV=[2, 2, 2],
    PADDING_Q=[1, 1, 1],
    STRIDE_Q=[1, 1, 1])
        model   = ConvolutionalVisionTransformer(num_classes=self.num_classes, spec=spec)
        self.net = model

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片，和预测图片
    #---------------------------------------------------#
    def detect_image(self, image, name, count=False, name_classes=None):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        # 上下文管理器，用于告诉PyTorch不要计算梯度
        # 所有涉及到的张量都不会计算梯度
        with torch.no_grad():
            # 将NumPy数组image_data转换为PyTorch张量
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
        
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print(name + '.png' + '\n')
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(0, self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
    
        if self.mix_type == 0:
            seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # *****************************************************************************************************************
            for c in range(1, self.num_classes):
                seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            # a = np.array(self.colors, np.uint8)[np.reshape(pr, [-1])]
            # print(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])])
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   将新图与原图及进行混合
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            for c in range(1, self.num_classes):
                seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            # print(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])])
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
        
        return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------#
                #   图片传入网络进行预测
                #---------------------------------------------------#
                pr = self.net(images)[0]
                #---------------------------------------------------#
                #   取出每一个像素点的种类
                #---------------------------------------------------#
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                #--------------------------------------#
                #   将灰条部分截取掉
                #--------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
    
    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)

            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr == c) * (self.colors[c][0])).astype('uint8')
            #     seg_img[:, :, 1] += ((pr == c) * (self.colors[c][1])).astype('uint8')
            #     seg_img[:, :, 2] += ((pr == c) * (self.colors[c][2])).astype('uint8')

        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        # image = Image.fromarray(np.uint8(seg_img))
        # ------------------------------------------------#
        #   将新图与原图及进行混合
        # ------------------------------------------------#
        # image = Image.blend(old_img, image, 0.7)
    
        image = Image.fromarray(np.uint8(pr))
        return image
