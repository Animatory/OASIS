import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations import Compose, RandomResizedCrop, HorizontalFlip, Normalize, \
    ShiftScaleRotate, CenterCrop, SmallestMaxSize
from albumentations.pytorch import ToTensorV2


class Ade20kDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.label_nc = 150
        opt.contain_dontcare_label = True
        opt.semantic_nc = 151  # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels = self.list_images()
        self.has_unsup_images = not (self.opt.unsup_dir is None or self.opt.phase == "test" or self.for_metrics)
        if self.has_unsup_images:
            self.unsup_images = sorted(Path(self.opt.unsup_dir).glob('**/*'))

        transforms_list = []
        if not (self.opt.phase == "test" or self.for_metrics):
            transforms_list.extend([
                HorizontalFlip(),
                RandomResizedCrop(opt.image_size, opt.image_size, scale=(0.6, 1.), ratio=(0.9, 1 / 0.9)),
                ShiftScaleRotate(rotate_limit=10),
            ])
        else:
            transforms_list.extend([
                SmallestMaxSize(opt.image_size),
                CenterCrop(opt.image_size, opt.image_size)
            ])
        transforms_list.extend([
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
        self.transforms = Compose(transforms_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert('RGB'))
        label = np.array(Image.open(self.labels[idx]))
        # assert image.shape[:2] == label.shape[:2]
        if image.shape[:2] != label.shape[:2]:
            image = cv2.resize(image, label.shape[::-1])
        data = self.transforms(image=image, mask=label)
        sample = {"image": data['image'], "label": data['mask'].long(), "name": self.images[idx].name}

        if self.has_unsup_images:
            path = random.choice(self.unsup_images)
            image_unsup = np.array(Image.open(path).convert('RGB'))
            sample['image_unsup'] = self.transforms(image=image_unsup)['image']
        return sample

    def list_images(self):
        dataroot = self.opt.dataroot
        mode = "validation" if self.opt.phase == "test" or self.for_metrics else "training"
        dataroot_img = dataroot / 'images' / mode
        dataroot_ann = dataroot / 'annotations' / mode
        images = sorted(dataroot_img.glob('**/*.jpg'))
        labels = sorted(dataroot_ann.glob('**/*.png'))

        data = (dataroot / 'sceneCategories.txt').read_text().splitlines()
        images_scenes = pd.DataFrame([i.split() for i in data], columns=['name', 'scene'])
        counts = images_scenes.scene.value_counts()
        appropriate = ['office', 'kitchen', 'corridor', 'home_office', 'attic', 'parlor', 'closet', 'nursery', 'lobby',
                       'warehouse_indoor']
        names = counts[pd.Series(counts.index).apply(lambda x: 'room' in x or x in appropriate).values].index
        appropriate = set(images_scenes[images_scenes.scene.isin(names)].name)
        images = [i for i in images if i.stem in appropriate]
        labels = [i for i in labels if i.stem in appropriate]

        assert len(images) == len(labels), f"different len of images and labels {len(images)} - {len(labels)}"
        for image_path, label_path in zip(images, labels):
            assert image_path.stem == label_path.stem, f'{image_path} and {label_path} don not match'
        return images, labels
