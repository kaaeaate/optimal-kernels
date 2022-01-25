import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss_binary(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += (1. - dice.data.cpu().numpy()) * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def calc_loss_multiclass(pred, target, metrics, bce_weight=0.5):
    loss_lst = []
    dice_lst = []
    for i in range(pred.shape[1]):
        bce = F.binary_cross_entropy_with_logits(pred[:, i].unsqueeze(1), 
                                                 target[:, i].unsqueeze(1))

        pred_new = F.sigmoid(pred[:, i].unsqueeze(1))
        dice = dice_loss(pred_new, target[:, i].unsqueeze(1))
        metrics[f'dice_{i}'] += (1. - dice.data.cpu().numpy()) * target.size(0)
    
        loss_cl = bce * bce_weight + dice * (1 - bce_weight)
        loss_lst.append(loss_cl)
        dice_lst.append(dice)
    loss = sum(loss_lst) / 3
    dice_full = sum(dice_lst) / 3
    
#     metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += (1. - dice_full.data.cpu().numpy()) * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def calc_loss(pred, target, metrics, bce_weight=0.5, regime='multiclass'):
    if regime == 'multiclass':
        loss  = calc_loss_multiclass(pred, target, metrics, bce_weight=0.5)
    else:
        loss  = calc_loss_binary(pred, target, metrics, bce_weight=0.5)
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))  
    
    
