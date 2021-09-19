import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import os
import albumentations as A
import cv2
from torchvision.transforms import ToTensor


class Birds_Dataset(Dataset):
    def __init__(self, 
                 images_folder,
                 masks_folder,
                 img_transform=None, masks_transform=None):
        super(Dataset, self).__init__()
        
        def find_names_in_fold(prefix):
            images_names = np.sort(os.listdir(prefix))
            list_names = np.sort(os.listdir(prefix / images_names[0])).tolist()
            for i, x in enumerate(list_names):
                list_names[i] = os.path.join(images_names[0],x)
            for i in images_names[1:]:
                list_names_onefold = np.sort(os.listdir(prefix / i)).tolist()
                for j, x in enumerate(list_names_onefold):
                    list_names_onefold[j] = os.path.join(i, x)
                list_names.extend(list_names_onefold)
            return list_names
        
        self.images_names = find_names_in_fold(prefix = images_folder)[::50]
        self.masks_names = find_names_in_fold(prefix = masks_folder)[::50]
        
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.img_transform = img_transform
        self.masks_transform = masks_transform

    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):

        item_image = Image.open(os.path.join(self.images_folder,
                                            self.images_names[idx])).convert('RGB')
        item_mask = Image.open(os.path.join(self.masks_folder,
                                              self.masks_names[idx])).convert('RGB')
        
        
        SEED = np.random.randint(123456789)
        if self.img_transform is not None:
            random.seed(SEED)
            item_image = self.img_transform(item_image)
        if self.masks_transform is not None:  
            random.seed(SEED)
            item_mask = self.masks_transform(item_mask)

        return item_image, item_mask


class ShapeDataset(Dataset):
    def __init__(self, images_folder, masks_folder, 
                 img_transform=None, masks_transform=None):
        super(Dataset, self).__init__()
        
        self.images_folder = images_folder
        self.masks_folder = masks_folder

        self.images_names = np.sort(os.listdir(images_folder))[::10] 
        self.masks_names = np.sort(os.listdir(masks_folder))[::10]  
        
        self.img_transform = img_transform
        self.masks_transform = masks_transform

    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):
        item_image = Image.open(os.path.join(self.images_folder,
                                            self.images_names[idx])).convert('RGB')
        item_mask = Image.open(os.path.join(self.masks_folder,
                                              self.masks_names[idx])).convert('RGB')
        
        SEED = np.random.randint(123456789)
        if self.img_transform is not None:
            random.seed(SEED)
            item_image = self.img_transform(item_image)
        if self.masks_transform is not None:  
            random.seed(SEED)
            item_mask = self.masks_transform(item_mask)

        return item_image, item_mask
    
    
    
class OneImageDataset(Dataset):
    def __init__(self, images_folder, masks_folder, 
                 idx,
                 img_transform=None):
        super(Dataset, self).__init__()
        
        self.images_folder = images_folder
        self.masks_folder = masks_folder

        self.images_names = np.sort(os.listdir(images_folder))[idx:idx+1] 
        self.masks_names = np.sort(os.listdir(masks_folder))[idx:idx+1]
        
        self.img_transform = img_transform
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):
#         item_image = Image.open(os.path.join(self.images_folder,
#                                             self.images_names[idx])).convert('RGB')
        
        item_image = cv2.imread(os.path.join(self.images_folder,
                                             self.images_names[idx]))
        item_image = cv2.cvtColor(item_image, cv2.COLOR_BGR2RGB)
        
        item_mask = cv2.imread(os.path.join(self.masks_folder,
                                             self.masks_names[idx]))
        item_mask = cv2.cvtColor(item_mask, cv2.COLOR_BGR2RGB)
        
        SEED = np.random.randint(123456789)
        if self.img_transform is not None:
            random.seed(SEED)
            transformed = self.img_transform(image=item_image, mask=item_mask)
            item_image = transformed['image']
            item_image = self.to_tensor(item_image.copy())
#             item_image = self.img_transform(item_image)

        return item_image