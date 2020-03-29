import os
from io import StringIO
import re

import torch
import pandas as pd
from skimage import io, transform
from skimage.color import gray2rgb
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

class FashionProductImages(Dataset):
    def __init__(self, cfg, split, logger=None, random_state=1234):
        """
        For trasnfer learning task, data with even years will be used as training set,
        those with odd years will be used as the test set.

        Top 20 classes is used for pre-training, the rest is for fine-tuning

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cfg = cfg.clone()
        self.logger = logger
        self.rootdir = self.cfg.DATALOADER.FP_DATASET.ROOTDIR
        self.topk = self.cfg.DATALOADER.FP_DATASET.TOPK
        self.bottomk = self.cfg.DATALOADER.FP_DATASET.BOTTOMK

        self.meta = self._read_csv(os.path.join(self.rootdir, 'styles.csv'))
        self.imgroot = os.path.join(self.rootdir, 'images')
        assert os.path.exists(self.imgroot)

        self.train_meta = None
        self.val_meta = None
        self.test_meta = None
        self.val_frac = self.cfg.DATALOADER.FP_DATASET.VAL_DATA_FRACTION

        if split == TRAIN:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize(cfg.INPUT.MIN_SIZE),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                                 ])
        else:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize(cfg.INPUT.MIN_SIZE),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                                 ])

        self.split = split    # 'train', 'val', 'test'
        self.random_state = random_state

        self._prepare_meta_split()
        self.params = self._set_params()


    def _set_params(self):
        params = {'batch_size': self.cfg.TRAIN.BATCH_SIZE,
                  'shuffle': self.cfg.TRAIN.SHUFFLE,
                  'num_workers': self.cfg.DATALOADER.NUM_WORKERS}
        return params


    def __len__(self):
        return len(self.meta)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id = self.meta.iloc[idx]['id']
        img_name = os.path.join(self.imgroot, '{}.jpg'.format(id))
        image = io.imread(img_name)

        if image.ndim == 2:
            image = gray2rgb(image)

        if self.transform:
            image = self.transform(image)

        sample = {col: self.meta.iloc[idx][col] for col in self.meta.columns}
        sample.update({'image': image,
                       'label_articleType': self.target_classes[sample['articleType']]})

        return sample


    def _read_csv(self, csv_file, num_commas=9):
        fp = StringIO()
        self.logger.info('reading annotation file [{}]'.format(csv_file))
        with open(csv_file) as lines:
            for line in lines:
                new_line = re.sub(r',', '|', line.rstrip(), count=num_commas)
                print(new_line, file=fp)

        fp.seek(0)

        return pd.read_csv(fp, sep='|')


    def _get_top_classes_order(self, print_k=20):
        '''

        :return: dict of (class name: frequency) as keys, values sorted by freq. Either top K classes or
                bottom K classes are returned
        '''
        assert not (self.topk > 0 and self.bottomk > 0)
        assert self.topk > 0 or self.bottomk > 0

        if self.topk < 0:
            stats = self.meta.groupby('articleType')['id'].count().sort_values(ascending=True)[0:self.bottomk].to_dict()
        else:
            stats = self.meta.groupby('articleType')['id'].count().sort_values(ascending=False)[0:self.topk].to_dict()

        if self.split == TRAIN:
            for i, k in enumerate(stats):
                print('{:02d}\t{}\t{}'.format(i, k, stats[k]))
                if i == print_k - 1:
                    break

        return stats


    def _prepare_meta_split(self):
        split = self.split
        # `stats` contains all classes of interest
        stats = self._get_top_classes_order()
        target_classes = list(stats.keys())

        self.logger.info('preparing {} data...'.format(split))
        if split == TRAIN or split == VAL:
            meta = self.meta.loc[self.meta['year'] % 2 == 0]    # even year
            meta = meta[meta['articleType'].isin(target_classes)]
            train_meta, val_meta = train_test_split(meta, test_size=self.val_frac, shuffle=True,
                                                    random_state=self.random_state)
            if split == TRAIN:
                self.meta = train_meta
            else:
                self.meta = val_meta
        elif split == TEST:
            meta = self.meta.loc[self.meta['year'] % 2 == 1]    # odd year
            self.meta = meta[meta['articleType'].isin(target_classes)]
        else:
            raise NotImplementedError

        self.target_classes = {cls: i for i, cls in enumerate(target_classes)}
        if self.cfg.DATALOADER.DEBUG:
            self.meta = self.meta[0:100]
