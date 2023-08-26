import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import os
import albumentations as A
import cv2
from torchvision.transforms import ToTensor

from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path


def multiclass_to_binary(mask, class_num=3):
    device = mask.device
    new_mask = torch.zeros((class_num, mask.shape[-2], mask.shape[-1]))
    new_mask = new_mask.to(device)
    for i in range(class_num):
        new_mask[i,:,:] = torch.where(mask==i+1, 1, 0)
    return new_mask

def get_lips_twins(images_path, masks_path):
    imgs = sorted(os.listdir(images_path))
    masks = sorted(os.listdir(masks_path))
    imgs_cut = [img.split('.')[0][-8:] for img in imgs]
    masks_cut = [mask.split('.')[0][-8:] for mask in masks]
    twins_idxs = sorted(list(set(imgs_cut).intersection(masks_cut)))
    images = sorted([image for image in imgs if image.split('.')[0][-8:] in twins_idxs])
    masks = sorted([mask for mask in masks if mask.split('.')[0][-8:] in twins_idxs])
    return images, masks

def get_train_test_dataset(dataset_path, images_path, masks_path,
                           dataset_dir, train_fold, test_fold,
                           img_fold, masks_fold, test_size = 0.3):
        
    if dataset_path.split('/')[-1] == 'Lipstick':
        dataset_path = Path(dataset_path)
        images, masks = get_lips_twins(images_path = dataset_path / images_path,
                                       masks_path = dataset_path / masks_path)
    else:
        dataset_path = Path(dataset_path)
        images = sorted(os.listdir(dataset_path / images_path))
        masks = sorted(os.listdir(dataset_path / masks_path))
    
    train_x, test_x, train_y, test_y = train_test_split(images, masks, test_size=test_size)
    
    dataset_dir = dataset_path / dataset_dir
    dataset_dir.mkdir(exist_ok=True)
    train_dir = dataset_path / dataset_dir / train_fold
    train_dir.mkdir(exist_ok=True)
    val_dir = dataset_path / dataset_dir / test_fold
    val_dir.mkdir(exist_ok=True)
    
    train_dir_imgs = dataset_path / dataset_dir / train_fold / img_fold
    train_dir_imgs.mkdir(exist_ok=True)
    train_dir_masks = dataset_path / dataset_dir / train_fold / masks_fold
    train_dir_masks.mkdir(exist_ok=True)
    val_dir_imgs = dataset_path / dataset_dir / test_fold / img_fold
    val_dir_imgs.mkdir(exist_ok=True)
    val_dir_masks = dataset_path / dataset_dir / test_fold / masks_fold
    val_dir_masks.mkdir(exist_ok=True)
    
    for img in train_x:
        shutil.copyfile(dataset_path / images_path / img, train_dir_imgs / img)
    for mask in train_y:
        shutil.copyfile(dataset_path / masks_path / mask, train_dir_masks / mask)

    for img in test_x:
        shutil.copyfile(dataset_path / images_path / img, val_dir_imgs / img)
    for mask in test_y:
        shutil.copyfile(dataset_path / masks_path / mask, val_dir_masks / mask) 