import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)


    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

    temp_target = target.view(-1)

    # ignore the background
    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=0)(temp_inputs, temp_target)
    return CE_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2, epsilon=1e-9):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=0, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    pt = torch.clamp(pt, min=epsilon, max=1.0)
    
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.permute(0, 2, 3, 1).contiguous().view(n, -1, c),dim=-1)
    # print(temp_inputs.shape)
    temp_target = target.view(n, -1, ct)
    # print(temp_target.shape)

    #   ignore the background
    tp = torch.sum(temp_target[..., 1:] * temp_inputs[..., 1:], axis=[0,1])
    fp = torch.sum(temp_inputs[..., 1:]                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[..., 1:]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        print(f'iters:{iters}')
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        #print(f'lr : {lr}')
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        #print(f'warmup_total_iters:{warmup_total_iters}')
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        #print(f'warmup_lr_start:{warmup_lr_start}')
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        #print(f'no_aug_iter:{no_aug_iter}')
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    #print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    num_classes     = 4

    pretrained      = True

    model_path      = ''

    downsample_factor   = 8
    input_shape         = [256, 256]
    

    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 8

    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 12

    Freeze_Train        = False
    Init_lr             = 5e-6
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = 'cos'
    UnFreeze_flag = False
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
         print(f'epoch:{epoch}')

         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
         nbs             = 16
         lr_limit_max    = 5e-6 if optimizer_type == 'adam' else 1e-1
         lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4

         Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
         print(f'Init_lr_fit:{Init_lr_fit}')
         Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
 
         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
         set_optimizer_lr(lr_scheduler_func, epoch)