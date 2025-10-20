import os, gc, random
import pandas as pd
import nibabel as nib
import torch
import monai
from monai import transforms

class aedataset(torch.utils.data.Dataset):
    def __init__(self, datafile, image_size, return_affine=False, return_img_name=False):
        
        """
        Args:
            datafile: the datafile specifying the location of images.
            image_size: desired size to resize input images to for compatibility with model architecture
            transforms (type: pytorch specific transforms): to add channel to the image and convert to tensor.
        Returns:
            img [torch tensor]: img file normalized 
            mask [torch tensor]: mask excluding background
            Also returns affine and image name if those parameters are set to True.
        """
        self.image_list = [line.replace('\n','') for line in open(datafile, 'r')]
        self.image_size = image_size
        self.transforms = transforms.Compose([
                        transforms.EnsureChannelFirst(channel_dim='no_channel'),
                        transforms.ResizeWithPadOrCrop(spatial_size=self.image_size), 
                        transforms.ToTensor()
                        ])
        self.return_affine = return_affine
        self.return_img_name = return_img_name

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idxx=int):
        img = nib.load(self.image_list[idxx])
        aff = img.affine
        img_name = os.path.basename(self.image_list[idxx])
        img = img.get_fdata()
        mask = img != 0
        #print(img.shape)
        mask = self.transforms(mask)
        img = (img - img[img != 0].mean()) / img[img != 0].std()
        img = self.transforms(img)

        img = img.type(torch.float)
        #mask = torch.tensor(mask)
        if self.return_affine and self.return_img_name:
            return img, mask, aff, img_name
        elif self.return_affine:
            return img, mask, aff
        elif self.return_img_name:
            return img, mask, img_name
        else:
            return img, mask

        #return img, mask

