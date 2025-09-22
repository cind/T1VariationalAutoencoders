#imports
import os, gc, random
import pandas as pd
import nibabel as nib
import torch
import monai
from monai import transforms

#defining transforms
transforms_monai = transforms.Compose([
                    transforms.AddChannel(),
                    #transforms.ResizeWithPadOrCrop(spatial_size=(184,224,184)), 
                    #transforms.ScaleIntensity(minv=0.0, maxv=1.0),
                    transforms.ToTensor()])

class aedataset(torch.utils.data.Dataset):
    def __init__(self, datafile, transforms=transforms_monai, return_affine=False, return_img_name=False):
        
        """
        Args:
            datafile (type: csv): the datafile mentioning the location of images.
            transforms (type: pytorch specific transforms): to add channel to the image and convert to tensor.
        Returns:
            img [torch tensor]: img file normalized 
            mask [torch tensor]: mask excluding background
        """
        self.image_list = [line.replace('\n','') for line in open(datafile, 'r')]
        #for i in self.image_list:
        #    i = i.replace('regtoMNI', 'zscore_norm')
        self.transforms = transforms
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

