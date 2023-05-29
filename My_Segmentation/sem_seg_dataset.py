#%%
from torch.utils.data import Dataset
import os
from pathlib import Path
import cv2
# %%
class SegmentationDataset(Dataset):
    def __init__(self,path_name) -> None:
        super().__init__()
        self.image_names = os.listdir(f"{path_name}/images")
        self.mask_names = os.listdir(f"{path_name}/masks")
        self.image_names = [f"{path_name}/images/{name}" for name in self.image_names]
        self.mask_names = [f"{path_name}/masks/{name}" for name in self.mask_names]

        self.img_stem = [Path(name).stem for name in self.image_names]
        self.msk_stem = [Path(name).stem for name in self.mask_names]
        self.img_msk_stem = set(self.img_stem) & set(self.msk_stem)
        self.image_paths = [name for name in self.image_names if Path(name).stem in self.img_msk_stem]
        self.mask_paths = [name for name in self.mask_names if Path(name).stem in self.img_msk_stem]


    def __len__(self):
        return len(self.img_msk_stem)
    
    def convert_mask(self,mask):
        # unlabeled
        mask[mask==155] = 0
        # building
        mask[mask==44] = 1
        # land
        mask[mask==91] = 2
        # water
        mask[mask==171] = 3
        # unlabeled
        mask[mask==172] = 4
        # building
        mask[mask==212] = 5

        return mask

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)
        mask = cv2.imread(self.mask_paths[index],cv2.IMREAD_GRAYSCALE)
        mask = self.convert_mask(mask)
        return image,mask


