import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from pathlib import Path

import sys
sys.path.append('../')
from models import unet, attention_unet, lorck, unet_deform_convs, unet_init
from utils.dataloaders import Birds_Dataset, Birds_OneCluster, ShapeDataset
from utils.dataloaders import MNISTBinarDataset, PancreasDataset, LipstickDataset

def get_model(model_name, in_chs=3, out_chs=3, k_size=15):
    
    if model_name.lower() == 'unet':
        model = unet.UNet(in_chs,out_chs)
    elif model_name.lower() == 'attention-unet':
        model = attention_unet.AttU_Net(in_chs,out_chs)
    elif model_name.lower() == 'lorck':
        model = lorck.LORCK(in_chs,out_chs)
    elif model_name.lower() == 'def-convs':
        model = unet_deform_convs.UNet(in_chs,out_chs)
    elif model_name.lower() == 'unet_init':
        model = unet_init.UNet(in_chs,out_chs,k_size=k_size)
    else:
        print(f'Not implemented {model_name} model')        
    return model
    

def get_dataloader(*args):
    dataset_name = args.dataset_name
    dataset_path = Path(args.dataset_path)   
    
    trans = A.Compose([
          A.RandomCrop(width=544, height=544),
          A.RandomRotate90(p=1),
          A.HorizontalFlip(p=0.5),
    #       A.VerticalFlip(p=0.5),
          A.RandomBrightnessContrast(p=0.8),
                      ])
    train_set = dataset_dct[dataset_name](images_folder = dataset_path / 'dataset/train/imgs', 
                                          masks_folder = dataset_path / 'dataset/train/masks',
                                          transform = trans)

    val_set = dataset_dct[dataset_name](images_folder = dataset_path / 'dataset/val/imgs', 
                                        masks_folder = dataset_path / 'dataset/val/masks',
                                        transform = trans)
    image_datasets = {
    'train': train_set , 'val': val_set
    }

    batch_size_train = args.batch_size
    batch_size_val = batch_size_train

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size_val, shuffle=True, num_workers=0)
    }
    return dataloaders


dataset_dct = {
    'birds': Birds_Dataset,
    'birds-cluster': Birds_OneCluster,
    'simple-shapes': ShapeDataset,
    'mnist-binar': MNISTBinarDataset,
    'pancreas': PancreasDataset,
    'lipstick': LipstickDataset
}