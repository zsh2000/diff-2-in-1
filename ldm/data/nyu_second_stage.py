from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
import os
import glob
import numpy as np
from PIL import Image
import cv2


def normalize(v):
    return v / np.linalg.norm(v)

class NYU_Dataset(Dataset):
    def __init__(
            self,
            split="train",
    ):
        self.root_dir = "./dataset/nyu_nyu_T600/"
        self.split = split
        self.imgs_folder_name = "img"
        self.sn_folder_name = "norm"
        self.caption_folder_name = "caption"

        self.downsample = 1
        self.max_len = -1
        self.img_wh = (int(512 * self.downsample), int(512 * self.downsample))

        self.define_transforms()

        self.build_metas()

    def define_transforms(self):
        self.transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

        self.transform_sn = T.Compose(
            [
                T.ToTensor(),
            ]
        )

    def build_metas(self):
        self.image_paths = []
        self.sn_paths = []

        if self.split == "train":
            self.imgs_folder_name = "train/img"
            self.sn_folder_name = "train/norm"
            self.caption_folder_name = "train/caption"

        elif self.split == "val":
            self.imgs_folder_name = "test/img"
            self.sn_folder_name = "test/norm"
            self.caption_folder_name = "test/caption"

        elif self.split == "test":
            self.imgs_folder_name = "test/img"
            self.sn_folder_name = "test/norm"
            self.caption_folder_name = "test/caption"


        self.image_paths = sorted(
            glob.glob(os.path.join(self.root_dir, self.imgs_folder_name, "*.png"))
        )

        self.sn_paths = sorted(
            glob.glob(os.path.join(self.root_dir, self.sn_folder_name, "*.png"))
        )

        self.caption_paths = sorted(
            glob.glob(os.path.join(self.root_dir, self.caption_folder_name, "*.txt"))
        )


    def __len__(self):
        print(len(self.image_paths))
        return len(self.image_paths) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        w, h = self.img_wh
        img_filename = self.image_paths[idx]

        img_base_name = os.path.basename(img_filename)

        if img_base_name[0] == "r":
            is_labelled = 0
        else:
            is_labelled = 1

        im_cv = cv2.imread(img_filename)
        img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB) / 255.

        img = cv2.resize(img, (512, 512))
        img = self.transform(img)
        img = (img * 2) - 1

        sn_filename = self.sn_paths[idx]
        sn_cv = cv2.imread(sn_filename)

        sn = cv2.cvtColor(sn_cv, cv2.COLOR_BGR2RGB) / 255.

        mask = np.sum(sn, axis=2, keepdims=True)
        mask = np.where(mask == 0, 0.0, 1.0)

        sn = cv2.resize(sn, (512, 512))
        mask = cv2.resize(mask, (512, 512))
        mask = (mask == 1)

        mask = np.expand_dims(mask, 2)
        sn = self.transform_sn(sn)
        sn = (sn * 2) - 1

        caption_filename = self.caption_paths[idx]
        with open(caption_filename, "r") as f:
            caption = f.read()

        sample = {}
        sample["image"] = img

        sample["sns"] = sn
        sample["sn_masks"] = mask
        sample["is_labelled"] = is_labelled
        sample["caption"] = caption
        return sample
