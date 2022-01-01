import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import os
import albumentations as A
import cv2
from torchvision.transforms import ToTensor
import albumentations as A

from utils.data_preprocessing import get_lips_twins, get_train_test_dataset, multiclass_to_binary

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
        
        self.images_names = find_names_in_fold(prefix = images_folder) #[::50]
        self.masks_names = find_names_in_fold(prefix = masks_folder) #[::50]
        
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
    
    
class Birds_OneCluster(Dataset):
    def __init__(self, 
                 img_names, 
                 mask_names,
                 images_folder,
                 masks_folder,
                 clusters, 
                 cluster_num, 
                 cluster_epoch, 
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
        
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        
        if img_names is not None:
            self.images_names = img_names
        else:
            self.images_names = find_names_in_fold(prefix = self.images_folder) 
        if mask_names is not None:
            self.masks_names = mask_names
        else:
            self.masks_names = find_names_in_fold(prefix = self.masks_folder) 
        if clusters is not None:   
            self.cluster_idxs = clusters[cluster_epoch][cluster_num]
            self.cluster_imgs = [self.images_names[i] for i in self.cluster_idxs]
            self.cluster_masks = [self.masks_names[i] for i in self.cluster_idxs]
        else:
            self.cluster_imgs = self.images_names
            self.cluster_masks = self.masks_names
        
        self.img_transform = img_transform
        self.masks_transform = masks_transform

    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):

        item_image = Image.open(os.path.join(self.images_folder,
                                            self.cluster_imgs[idx])).convert('RGB')
        item_mask = Image.open(os.path.join(self.masks_folder,
                                              self.cluster_masks[idx])).convert('RGB')
        
        
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

        self.images_names = np.sort(os.listdir(images_folder))#[::10] 
        self.masks_names = np.sort(os.listdir(masks_folder))#[::10]  
        
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


class MNISTBinarDataset(Dataset):
    def __init__(self, images_folder, masks_folder, 
                 img_transform=None, masks_transform=None):
        super(Dataset, self).__init__()
        
        self.images_folder = images_folder
        self.masks_folder = masks_folder

        self.images_names = np.sort(os.listdir(images_folder))#[::5] 
        self.masks_names = np.sort(os.listdir(masks_folder))#[::5]  
        
        self.img_transform = img_transform
        self.masks_transform = masks_transform

    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):
        item_image = np.load(os.path.join(self.images_folder,
                                            self.images_names[idx]))
        item_mask = np.load(os.path.join(self.masks_folder,
                                              self.masks_names[idx]))
        
        SEED = np.random.randint(123456789)
        if self.img_transform is not None:
            random.seed(SEED)
            item_image = self.img_transform(item_image)
        if self.masks_transform is not None:  
            random.seed(SEED)
            item_mask = self.masks_transform(item_mask)

        return item_image, item_mask

    
class PancreasDataset(Dataset):
    def __init__(self, images_folder, masks_folder, 
                 img_transform=None, masks_transform=None):
        super(Dataset, self).__init__()
        
        self.images_folder = images_folder
        self.masks_folder = masks_folder

        self.images_names = np.sort(os.listdir(images_folder))
        self.masks_names = np.sort(os.listdir(masks_folder))
        
        self.img_transform = img_transform
        self.masks_transform = masks_transform

    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):
        item_image = Image.open(os.path.join(self.images_folder,
                                            self.images_names[idx])).convert('RGB')
        item_mask = cv2.imread(os.path.join(self.masks_folder,
                                              self.masks_names[idx]))
        
        item_mask2 = np.rot90(np.fliplr(item_mask),2)
        item_mask = np.where(item_mask2>0, 1, 0)
        
        SEED = np.random.randint(123456789)
        if self.img_transform is not None:
            random.seed(SEED)
            item_image = self.img_transform(item_image)
        if self.masks_transform is not None:  
            random.seed(SEED)
            item_mask = self.masks_transform(item_mask)

        return item_image, item_mask
    
    
class LipstickDataset(Dataset):
    def __init__(self, images_folder, masks_folder, 
                 transform=None):
        super(Dataset, self).__init__()
        
        self.images_folder = images_folder
        self.masks_folder = masks_folder

        self.images_names = np.sort(os.listdir(images_folder))#[::100]
        self.masks_names = np.sort(os.listdir(masks_folder))#[::100] 
        
        self.transform = transform
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):
        
        item_image = np.asarray(Image.open(os.path.join(self.images_folder,
                                            self.images_names[idx])))
        item_mask = np.asarray(Image.open(os.path.join(self.masks_folder,
                                              self.masks_names[idx])))[:,:,1]          

        if self.transform is not None:
            transformed = self.transform(image=item_image, mask=item_mask)
            item_image = transformed["image"]
            item_mask = transformed["mask"]

        item_image = self.to_tensor(item_image.copy())
        item_mask = self.to_tensor(item_mask.copy()) #torch.from_numpy(item_mask.copy()).long()

        return item_image, item_mask    
    

class MDSDataset(Dataset):
    def __init__(self, images_folder, masks_folder, 
                 transform=None,
                 classes=2):
        super(Dataset, self).__init__()
        
        self.images_folder = images_folder
        self.masks_folder = masks_folder

        self.images_names = np.sort(os.listdir(images_folder)) 
        self.masks_names = np.sort(os.listdir(masks_folder))  
        
        self.transform = transform
        self.to_tensor = ToTensor()
        self.classes = classes

    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):
        item_image = np.load(os.path.join(self.images_folder,
                                            self.images_names[idx]))
        item_mask = np.load(os.path.join(self.masks_folder,
                                              self.masks_names[idx]))
        
        if self.transform is not None:
            transformed = self.transform(image=item_image, mask=item_mask)
            item_image = transformed["image"]
            item_mask = transformed["mask"]

        item_image = self.to_tensor(item_image.copy())
        item_mask = self.to_tensor(item_mask.copy())
        if self.classes > 2:
            item_mask = multiclass_to_binary(item_mask, num_classes=self.classes)

        return item_image, item_mask
    
    
class ACDCDataset(torch.utils.data.Dataset):
    CLASSES = {0: 'NOR', 1: 'MINF', 2: 'DCM', 3: 'HCM', 4: 'RV'}

    def __init__(self, hf_path, device):
        super().__init__()
        self.device = device
        self.hf = h5py.File(hf_path)

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, item: int):
        img = self.hf[str(item)][:1]
        mask = self.hf[str(item)][1:]
        c = self.hf[str(item)].attrs['class']
        img = torch.tensor(img).float()
        mask = torch.tensor(mask)
        binary_mask = multiclass_to_binary(mask)
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-11)
        return dict(
            c=c, 
            mask=mask.to(self.device),
            binary_mask = binary_mask.to(self.device),
            img=img.to(self.device), 
            mean=mean[None, None, None].to(self.device), 
            std=std[None, None, None].to(self.device)
        )