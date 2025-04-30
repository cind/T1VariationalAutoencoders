#imports
import os, gc, random
import pandas as pd
import nibabel as nib
import torch
from monai import transforms

#defining transforms
transforms_monai = transforms.Compose([
                    transforms.ResizeWithPadOrCrop(spatial_size=(180,220,180)), 
                    transforms.AddChannel(), 
                    transforms.ToTensor()])

class aedataset(torch.utils.data.Dataset):
    def __init__(self, datafile, transforms=transforms_monai):
        
        """
        Args:
            datafile (type: csv): the datafile mentioning the location of images.
            transforms (type: pytorch specific transforms): to add channel to the image and convert to tensor.
        Returns:
            img [torch tensor]: img file normalized 
            mask [torch tensor]: mask excluding background
        """
        self.image_list = [line.replace('\n','') for line in open(datafile, 'r')]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idxx=int):
        img = nib.load(self.image_list[idxx])
        img = img.get_fdata()
        mask = img != 0
        img = (img - img[img != 0].mean()) / img[img != 0].std()
        img = self.transforms(img)

        img = img.type(torch.float)
        mask = torch.tensor(mask)

        return img, mask

