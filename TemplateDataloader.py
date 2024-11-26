import cv2
import glob
import os
import torch
import numpy as np
import torch.utils.data as data
from torchvision.io.image import read_image
from torchvision.transforms import v2
from torchvision import tv_tensors

mean = [0.46617496, 0.36034706, 0.33016744]
std = [0.23478602, 0.21425594, 0.20241965]

class SexBaseDataset(data.Dataset):
    def __init__(self, folder_path):
        super(SexBaseDataset, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'image','*.png'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'mask',os.path.basename(img_path)))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            #b,r,g => r,g,b => (1,2,0)
            data =  np.transpose(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (2, 0, 1))
            label = np.transpose(cv2.imread(mask_path), (2, 0, 1))
            return torch.from_numpy(data), torch.from_numpy(label)

    def __len__(self):
        return len(self.img_files)

#DataLoader = SexBaseDataset ("G:\\jav folder\\OutputFolder")

class SexDataset(SexBaseDataset):
    def __init__(self, folder_path, transforms=None, augmented = False):
        super().__init__(folder_path)
        self.transforms = transforms
        self.augmented = augmented    
        if self.transforms is None:
            self.transforms = v2.Compose([    
                #v2.ToDtype(torch.float32, scale=True), #Auto scale from 0-255 to 0.0-1.0
                v2.Normalize(mean=mean, std=std)
            ])
        if self.augmented == True:
            self.augmentPosition = v2.Compose([                
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.15),
                v2.RandomRotation(degrees = 6),
                v2.RandomAffine(degrees = 6),
                v2.RandomPerspective(distortion_scale=0.15, p=0.80)
            ])
            self.augmentColour = v2.Compose([
                v2.ColorJitter(brightness=0.5, contrast = 0.4, saturation = 0.2, hue = 0.04),
                v2.GaussianNoise(mean = 0, sigma = 0.015) #default sigma 0.1
            ])
    # Override the __getitem__ method
    def __getitem__(self, idx):
        # Call the parent class's __getitem__ to get the default behavior
        image, mask = super().__getitem__(idx)
        image = v2.functional.to_dtype (image, scale = True)
        if self.augmented == True:
            image = self.augmentColour(tv_tensors.Image(image))
            image, mask = self.augmentPosition (tv_tensors.Image(image), tv_tensors.Mask(mask)) 
        if self.transforms is not None:
            image = self.transforms(image)
        mask = mask[0,:,:]/255
        return image, mask
