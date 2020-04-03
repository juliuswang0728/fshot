import os
from io import StringIO
import re
import json

import torch
import pandas as pd
from skimage import io, transform
from skimage.color import gray2rgb
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

meta_train_targets = 'Cufflinks, Rompers, Laptop Bag, Sports Sandals, Hair Colour, Suspenders, Trousers, Kajal and Eyeliner, Compact, Concealer, Jackets, Mufflers, Backpacks, Sandals, Shorts, Waistcoat, Watches, Pendant, Basketballs, Bath Robe, Boxers, Deodorant, Rain Jacket, Necklace and Chains, Ring, Formal Shoes, Nail Polish, Baby Dolls, Lip Liner, Bangle, Tshirts, Flats, Stockings, Skirts, Mobile Pouch, Capris, Dupatta, Lip Gloss, Patiala, Handbags, Leggings, Ties, Flip Flops, Rucksacks, Jeggings, Nightdress, Waist Pouch, Tops, Dresses, Water Bottle, Camisoles, Heels, Gloves, Duffel Bag, Swimwear, Booties, Kurtis, Belts, Accessory Gift Set, Bra'
meta_test_targets = 'Jeans, Bracelet, Eyeshadow, Sweaters, Sarees, Earrings, Casual Shoes, Tracksuits, Clutches, Socks, Innerwear Vests, Night suits, Salwar, Stoles, Face Moisturisers, Perfume and Body Mist, Lounge Shorts, Scarves, Briefs, Jumpsuit, Wallets, Foundation and Primer, Sports Shoes, Highlighter and Blush, Sunscreen, Shoe Accessories, Track Pants, Fragrance Gift Set, Shirts, Sweatshirts, Mask and Peel, Jewellery Set, Face Wash and Cleanser, Messenger Bag, Free Gifts, Kurtas, Mascara, Lounge Pants, Caps, Lip Care, Trunk, Tunics, Kurta Sets, Sunglasses, Lipstick, Churidar, Travel Accessory'


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
        self.imgroot = os.path.join(self.rootdir, 'images')
        assert os.path.exists(self.imgroot)

        self.split = split  # 'train', 'val', 'test'
        self.meta = self._read_csv(os.path.join(self.rootdir, 'styles.csv'))
        self.train_meta = None
        self.val_meta = None
        self.test_meta = None
        self.val_frac = self.cfg.DATALOADER.FP_DATASET.VAL_DATA_FRACTION

        if split == TRAIN:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize((cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH)),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                                 ])
        else:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize((cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                                 ])

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

        image = self.transform(image)

        sample = {col: self.meta.iloc[idx][col] for col in self.meta.columns}
        #sample.update({'image': image,
        #               'label_articleType': self.target_classes[sample['articleType']]})
        sample = {'image': image, 'label_articleType': self.target_classes[sample['articleType']]}

        return sample


    def _read_csv(self, csv_file, num_commas=9):
        fp = StringIO()
        self.logger.info('[creating {} set] reading annotation file [{}]'.format(self.split, csv_file))
        with open(csv_file) as lines:
            for line in lines:
                new_line = re.sub(r',', '|', line.rstrip(), count=num_commas)
                print(new_line, file=fp)

        fp.seek(0)
        meta = pd.read_csv(fp, sep='|')

        self.logger.info('checking every image paths...')
        if os.path.exists(os.path.join(self.rootdir, 'images.csv')):
            images_csv = pd.read_csv(os.path.join(self.rootdir, 'images.csv'))
            drop_ind = images_csv.loc[images_csv['link'] == 'undefined'].index.tolist()
        else:
            self.logger.info('(SLOWER) no images.csv exists, literally check every path to the image')
            drop_ind = [i for i, id in enumerate(meta.loc[:, 'id'])
                        if not os.path.exists(os.path.join(self.imgroot, '{}.jpg'.format(id)))]

        if len(drop_ind) > 0:
            self.logger.info('dropping undefined images: {}'.format(drop_ind))
            meta = meta.drop(drop_ind)

        return meta


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

        stats = self.meta.groupby('articleType')['id'].count().sort_values(ascending=False)[0:self.topk].to_dict()
        self.num_images = sum(stats.values())
        self.logger.info('data stats:\n{}'.format(json.dumps(stats)))
        self.logger.info('number of images: {}'.format(self.num_images))


class FashionProductImagesFewShot(Dataset):
    def __init__(self, cfg, split, k_way, n_shot, logger=None, random_state=1234):
        assert split in [TRAIN, TEST]
        self.split = split
        self.cfg = cfg.clone()
        self.logger = logger
        self.k_way = k_way
        self.n_shot = n_shot
        self.rootdir = self.cfg.DATALOADER.FP_DATASET.ROOTDIR
        self.imgroot = os.path.join(self.rootdir, 'images')
        assert os.path.exists(self.imgroot)
        self.meta_index2class = self._get_meta_index2class()
        self.meta_class2index = {cat: index for index, cat in self.meta_index2class.items()}
        self.meta = self._read_csv(os.path.join(self.rootdir, 'styles.csv'))

        self.train_meta = None
        self.test_meta = None

        self.random_state = random_state

        self._prepare_meta_split()
        self.params = self._set_params()

        if split == TRAIN:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize((cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH)),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                                 ])
        if split == TEST:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize((cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                                 ])

        self.get_item = self._get_meta_train if split == TRAIN else self._get_meta_test


    def _get_meta_index2class(self):
        def _get_index2class(targets):
            return {i: name for i, name in enumerate(targets)}
        if self.split == TRAIN:
            return _get_index2class(meta_train_targets.split(', '))
        if self.split == TEST:
            return _get_index2class(meta_test_targets.split(', '))


    def _set_params(self):
        assert self.cfg.TRAIN.BATCH_SIZE == 1, "TRAIN.BATCH_SIZE should be set to 1 in few-shot meta-learning/test phases"
        params = {'batch_size': self.cfg.TRAIN.BATCH_SIZE,
                  'shuffle': self.cfg.TRAIN.SHUFFLE,
                  'num_workers': self.cfg.DATALOADER.NUM_WORKERS}
        return params


    def __len__(self):
        return len(self.meta)


    def _get_images(self, ind):
        get_img_path = lambda id: os.path.join(self.imgroot, '{}.jpg'.format(id))

        def _get_image(id):
            path = get_img_path(id)
            if not os.path.exists(path):
                print('non-exist path: {}'.format(path))
            image = io.imread(path)
            if image.ndim == 2:
                return self.transform(gray2rgb(image))
            return self.transform(image)

        return [_get_image(i) for i in self.meta.iloc[ind]['id'].to_list()]


    def _get_meta_train(self):
        candidate_classes = list(range(len(self.meta_index2class)))
        sampled_classes = random.sample(candidate_classes, self.k_way)
        labels = sampled_classes
        meta_set = {cls: [] for cls in sampled_classes}
        samples = []
        # sample n_shot * k_way samples from those in the sampled classes
        for cls in sampled_classes:
            meta_samples = random.sample(self.sample_ind_per_class[self.meta_index2class[cls]], self.n_shot)
            meta_set[cls] += meta_samples
            samples += meta_samples

        # sample a query sample from each of the sampled classes
        for cls in sampled_classes:
            while True:
                query_sample = random.sample(self.sample_ind_per_class[self.meta_index2class[cls]], 1)[0]
                if query_sample not in meta_set[cls]:
                    samples.append(query_sample)
                    break

        return samples, labels


    def _get_meta_test(self, idx):
        id, query_class = self.meta.iloc[idx]['id'], self.meta.iloc[idx]['articleType']

        candidate_classes = list(range(len(self.meta_index2class)))
        candidate_classes.remove(self.meta_class2index[query_class])
        sampled_classes = random.sample(candidate_classes, self.k_way)
        label = np.random.randint(0, self.k_way)
        sampled_classes[label] = self.meta_class2index[query_class]

        meta_set = {cls: [] for cls in sampled_classes}
        samples = []
        # sample n_shot * k_way samples from those in the sampled classes
        for cls in sampled_classes:
            meta_samples = random.sample(self.sample_ind_per_class[self.meta_index2class[cls]], self.n_shot)
            if cls == label:
                while idx in meta_samples:
                    meta_samples.remove(idx)
                    meta_samples += random.sample(self.sample_ind_per_class[self.meta_index2class[cls]], 1)

            meta_set[cls] += meta_samples
            samples += meta_samples

        samples += [idx]    # add query sample to the end
        return samples, label


    def __getitem__(self, idx):
        '''
        :param idx: idx is here just for counting how many times the loader has been called to enumerate,
                    but not for indicating the idx of the sample picked each episode
        :return sample: return `images`: n_shot * k_way meta-train samples and k_way query samples, 1 sample each of the k-way classes
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.split == TRAIN:
            samples, labels = self.get_item()
        elif self.split == TEST:
            samples, labels = self.get_item(idx)

        images = self._get_images(samples)
        sample = {'images': images, 'labels': labels}

        return sample


    def _read_csv(self, csv_file, num_commas=9):
        fp = StringIO()
        self.logger.info('[creating {} set] reading annotation file [{}]'.format(self.split, csv_file))
        with open(csv_file) as lines:
            for line in lines:
                new_line = re.sub(r',', '|', line.rstrip(), count=num_commas)
                print(new_line, file=fp)

        fp.seek(0)
        meta = pd.read_csv(fp, sep='|')

        self.logger.info('checking every image paths...')
        if os.path.exists(os.path.join(self.rootdir, 'images.csv')):
            images_csv = pd.read_csv(os.path.join(self.rootdir, 'images.csv'))
            drop_ind = images_csv.loc[images_csv['link'] == 'undefined'].index.tolist()
        else:
            self.logger.info('(SLOWER) no images.csv exists, literally check every path to the image')
            drop_ind = [i for i, id in enumerate(meta.loc[:, 'id'])
                        if not os.path.exists(os.path.join(self.imgroot, '{}.jpg'.format(id)))]

        if len(drop_ind) > 0:
            self.logger.info('dropping undefined images: {}'.format(drop_ind))
            meta = meta.drop(drop_ind)

        return meta


    def _prepare_meta_split(self):
        self.meta = self.meta[self.meta['articleType'].isin(self.meta_index2class.values())].reset_index()

        stats = self.meta.groupby('articleType')['id'].count().sort_values(ascending=False).to_dict()
        self.num_images = sum(stats.values())
        self.logger.info('data stats:\n{}'.format(json.dumps(stats)))
        self.logger.info('number of images: {}'.format(self.num_images))
        self.sample_ind_per_class = self._samples_grouped_by_class()


    def _samples_grouped_by_class(self):
        # return indices where each index is used in xxx.loc[index]
        return self.meta.groupby('articleType').apply(lambda x: x.index.tolist())