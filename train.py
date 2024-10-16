import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

'''
    # The support packages required for the compare experiment.
    #
    from nets.decode_head import BaseDecodeHead
    from mmcv.cnn import ConvModule
    #
    from nets.unet import Unet
    from nets.pspnet import PSPNet
    from nets.deeplabv3_plus import DeepLab
    from nets.hrnet import HRnet
    from nets.CGNet import Context_Guided_Network
    from nets.segformer import se
    from nets.PVT_L import PVT_L
    model_path = "/yourdirs/model_data/pvt_v2_b4.pth"
    from nets.CVT import ConvolutionalVisionTransformer
    model_path = "/yourdirs/model_data/CvT-13-384x384-IN-1k.pth"
    from nets.DDRNet import DualResNet_imagenet
    from nets.seaformer import sa
    model_path = "/yourdirs/model_data/SeaFormer_L_cls_79.9.pth.tar"
    from nets.SCTNet import sct
'''
from nets.HSCRNet import HSCR
from nets.HSCR_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import HSCRDataset, dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
    U-Net:           Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone)
    PSPNet:          PSPNet(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained, aux_branch=aux_branch)
    Deeplabv3plus:   DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    HRNet：          HRnet(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    CGNet:           Context_Guided_Network(classes=num_classes)
    SegFormer:       se(num_classes=num_classes, input_size=input_shape)
    PVT:             PVT_L(num_classes=num_classes, input_size=input_shape)
    CVT:             ConvolutionalVisionTransformer(num_classes=num_classes, spec=spec)
    DDRNet:          DualResNet_imagenet(img_size=input_shape, num_classes=num_classes)
    SeaFormer:       sa(model_cfgs, num_classes=num_classes)
    SCTNet:          sct(num_classes=num_classes, input_size=input_shape)
'''

if __name__ == "__main__":
    
    #  configurations
    Cuda            = True
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = True
    num_classes     = 7
    backbone        = 'mobilenet'
    pretrained      = False
    model_path      = ''
    downsample_factor   = 8
    input_shape         = [256, 256]
    Init_Epoch          = 0
    Freeze_Epoch        = 60
    Freeze_batch_size   = 8
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 12
    Freeze_Train        = False
    Init_lr             = 5e-5
    Min_lr              = Init_lr * 0.01
    optimizer_type      = 'Adam'
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = 'cos'
    save_period         = 15
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 30
    DATASET_path        = 'WHDLD'#'landcover.ai'
    dice_loss       = True
    focal_loss      = True
    cls_weights     = np.ones([num_classes], np.float32)
    num_workers         = 16
    
    seed_everything(seed)
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)
    '''
    # CVT
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
            STRIDE_Q=[1, 1, 1]
            )
    
    # SeaFormer
    model_cfgs = dict(
            cfg1=[
                # k,  t,  c, s
                [3, 3, 32, 1],  
                [3, 4, 64, 2], 
                [3, 4, 64, 1]],  
            cfg2=[
                [5, 4, 128, 2],  
                [5, 4, 128, 1]],  
            cfg3=[
                [3, 4, 192, 2],  
                [3, 4, 192, 1]],
            cfg4=[
                [5, 4, 256, 2]],  
            cfg5=[
                [3, 6, 320, 2]], 
            channels=[32, 64, 128, 192, 256, 320],
            depths=[3, 3, 3],
            key_dims=[16, 20, 24],
            emb_dims=[192, 256, 320],
            num_heads=8,
            mlp_ratios=[2,4,6],
            drop_path_rate=0.1,
            norm_cfg = dict(type='BN', requires_grad=True)
            )
    '''
    model   = HSCR(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':

        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:

            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:

            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #  reading your data from .txt
    with open(os.path.join(DATASET_path, "segmentaion_sets/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(DATASET_path, "segmentaion_sets/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    print(num_val)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape,
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train,
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type,
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #   Adaptively adjust the learning rate based on the current batch_size.
        nbs             = 16
        lr_limit_max    = 5e-5 if optimizer_type == 'Adam' else 1e-1
        lr_limit_min    = 3e-5 if optimizer_type == 'Adam' else 5e-5
        if backbone == "xception":
            lr_limit_max    = 1e-4 if optimizer_type == 'Adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'Adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        if Init_lr             == 3.51e-5:
            Init_lr_fit = 3.51e-5

        #   Select the optimiser based on optimizer_type.
        optimizer = {
            'Adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #   Learning rate update formula.
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        

        train_dataset   = HSCRDataset(train_lines, input_shape, num_classes, True, DATASET_path)
        val_dataset     = HSCRDataset(val_lines, input_shape, num_classes, False, DATASET_path)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = dataset_collate, sampler=train_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = dataset_collate, sampler=val_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #   recording 
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, DATASET_path, log_dir, Cuda,
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #   training
        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 5e-5 if optimizer_type == 'Adam' else 1e-1
                lr_limit_min    = 3e-5 if optimizer_type == 'Adam' else 5e-5
                if backbone == "xception":
                    lr_limit_max    = 1e-4 if optimizer_type == 'Adam' else 1e-1
                    lr_limit_min    = 1e-4 if optimizer_type == 'Adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = dataset_collate, sampler=train_sampler,
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = dataset_collate, sampler=val_sampler,
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True


            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss,
                          focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
