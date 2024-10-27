import torch.utils.data as data
import cv2
import glob
import os
import torch
import numpy as np
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
