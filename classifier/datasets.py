import csv
import os
import random
import math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import namedtuple

from typing import Optional
from functools import partial
from collections import namedtuple
from torchvision.datasets.utils import verify_str_arg
import torchvision.transforms as transforms

CSV = namedtuple("CSV", ["header", "index", "data"])


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class Clevr2DPosDataset(Dataset):
    def __init__(
        self,
        resolution,
        random_crop=False,
        random_flip=False,
        split=None
    ):
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip

        data_path = 'clevr_pos_data_128_30000.npz'
        data = np.load(data_path)

        # split == None --> use the whole dataset
        self.ims, self.labels = data['ims'], data['coords_labels']

        N = self.ims.shape[0]
        if split == 'train':
            self.ims = self.ims[:int(N * 0.8)]
            self.labels = self.labels[:int(N * 0.8)]
        elif split == 'val':
            self.ims = self.ims[int(N * 0.8):]
            self.labels = self.labels[int(N * 0.8):]
        else:
            raise ValueError('Split needs to be specified.')

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.ims[index]).convert('RGB')
        pos = self.labels[index]
        label = 1

        if random.uniform(0, 1) < 0.5:
            if random.uniform(0, 1) < 0.5:
                # sample negative relation
                x = np.random.uniform(0, 1)
                y = np.random.uniform(0, 1)
                pos = np.array([x, y])
                label = 0
            else:
                # sample negative image
                neg_idx = random.randint(0, len(self.ims) - 1)
                while neg_idx == index or np.abs(np.sum(self.labels[neg_idx] - pos)) < 1e-5:
                    neg_idx = random.randint(0, len(self.ims) - 1)
                image = Image.fromarray(self.ims[neg_idx]).convert('RGB')
                label = 0

        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        # range 0 to 1
        arr = arr.astype(np.float32) / 255.
        return np.transpose(arr, [2, 0, 1]), pos, label


class FFHQSubdataset(Dataset):
    def __init__(
        self,
        resolution,
        random_crop=False,
        random_flip=False,
        split=None,
        cond_idx=0
    ):
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.split = split

        data_path = 'ffhq_256_dataset.npz'
        data = np.load(data_path)
        self.ims, self.labels = data['ims'], data['labels']

        N = self.ims.shape[0]
        np.random.seed(0)
        idx = np.arange(N)
        np.random.shuffle(idx)

        if split == 'train':
            self.ims = self.ims[idx][:int(N * 0.8)]
            self.labels = self.labels[idx][:int(N * 0.8)]
        elif split == 'val':
            self.ims = self.ims[idx][int(N * 0.8):]
            self.labels = self.labels[idx][int(N * 0.8):]
        else:
            raise ValueError('Split needs to be specified.')
        # cond_idx=0 --> smile
        # cond_idx=1 --> glasses
        # cond_idx=3 --> gender
        self.labels = self.labels[:, cond_idx]

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.ims[index]).convert('RGB')
        attr = self.labels[index]
        label = 1

        if random.uniform(0, 1) < 0.5 and self.split == 'train':
            if random.uniform(0, 1) < 0.5:
                # sample negative relation
                attr = 0 if attr == 1 else 1
                label = 0
            else:
                # sample negative image
                neg_idx = random.randint(0, len(self.ims) - 1)
                while neg_idx == index or self.labels[neg_idx] == self.labels[index]:
                    neg_idx = random.randint(0, len(self.ims) - 1)
                image = Image.fromarray(self.ims[neg_idx]).convert('RGB')
                label = 0

        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        # range 0 to 1
        arr = arr.astype(np.float32) / 255.
        return np.transpose(arr, [2, 0, 1]), attr, label


class ClevrRelDataset(Dataset):
    def __init__(
        self,
        resolution,
        random_crop=False,
        random_flip=False,
        split=None
    ):
        self.split = split
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip

        import pickle
        with open('clevr_rel_128_30000.pickle', 'rb') as f:
            data_dict = pickle.load(f)
            self.ims = data_dict['img']
            self.labels = data_dict['object_relations_labels']

        for i in range(len(self.labels)):
            self.labels[i] = np.array(self.labels[i])

        N = len(self.ims)
        if split == 'train':
            self.ims = self.ims[:int(N * 0.8)]
            self.labels = self.labels[:int(N * 0.8)]
        elif split == 'val':
            self.ims = self.ims[int(N * 0.8):]
            self.labels = self.labels[int(N * 0.8):]
        else:
            raise ValueError('Split needs to be specified.')

        Image.fromarray(self.ims[0]).save('sample.png')

        # caption mapping
        colors_to_idx = {"gray": 0, "red": 1, "blue": 2, "green": 3, "brown": 4, "purple": 5, "cyan": 6, "yellow": 7,
                         "none": 8}
        shapes_to_idx = {"cube": 0, "sphere": 1, "cylinder": 2, "none": 3}
        materials_to_idx = {"rubber": 0, "metal": 1, "none": 2}
        sizes_to_idx = {"small": 0, "large": 1, "none": 2}
        relations_to_idx = {"left": 0, "right": 1, "front": 2, "behind": 3, "below": 4, "above": 5, "none": 6}

        self.label_description = {
            "left": "to the left of",
            "right": "to the right of",
            "behind": "behind",
            "front": "in front of",
            "above": "above",
            "below": "below"
        }

        self.colors = list(colors_to_idx.keys())
        self.shapes = list(shapes_to_idx.keys())
        self.materials = list(materials_to_idx.keys())
        self.sizes = list(sizes_to_idx.keys())
        self.relations = list(relations_to_idx.keys())

    def random_object(self):
        shape = random.randint(0, len(self.shapes) - 2)
        size = random.randint(0, len(self.sizes) - 2)
        color = random.randint(0, len(self.colors) - 2)
        material = random.randint(0, len(self.materials) - 2)
        return [shape, size, color, material]

    def random_object_relation(self):
        obj1 = self.random_object()
        obj2 = self.random_object()
        relation = random.randint(0, len(self.relations) - 1)
        return np.array(obj1 + [0] + obj2 + [1] + [relation])

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, index):
        image = Image.fromarray(self.ims[index]).convert('RGB')
        assert self.labels[index].shape[0] > 0
        rel_idx = random.randint(0, self.labels[index].shape[0] - 1)
        relation = self.labels[index][rel_idx]
        label = 1

        if random.uniform(0, 1) < 0.5:
            if random.uniform(0, 1) < 0.5:
                # sample negative relation
                relation = self.random_object_relation()
                while (relation == self.labels[index]).all(axis=1).any():
                    relation = self.random_object_relation()
                label = 0
            else:
                # sample negative image
                neg_idx = random.randint(0, len(self.ims) - 1)
                while neg_idx == index or (relation == self.labels[neg_idx]).all(axis=1).any():
                    neg_idx = random.randint(0, len(self.ims) - 1)
                image = Image.fromarray(self.ims[neg_idx]).convert('RGB')
                label = 0

        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        # range 0 to 1
        arr = arr.astype(np.float32) / 255.

        return np.transpose(arr, [2, 0, 1]), relation, label


class CelebA128dataset(Dataset):
    def __init__(self,
                 root: str,
                 hq: bool = True,
                 split: str = "train",
                 attr_name: str = "smiling",
                 resolution: int = 128,
                 random_crop: bool = False,
                 random_flip: bool = False):
        super().__init__()
        self.root = root
        self.base_folder = 'celeba_hq' if hq else 'celeba'
        self.split = split
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip

        # load numpy data
        self.data_path = 'celeba_hq_data.npz'
        celeba_data = np.load(self.data_path)

        if self.split == 'train':
            self.ims = celeba_data['ims']
            self.labels = celeba_data['labels']
        elif self.split == 'val':
            self.ims = celeba_data['ims']
            self.labels = celeba_data['labels']
        else:
            self.ims = celeba_data['ims']
            self.labels = celeba_data['labels']

        if hq:
            N = self.ims.shape[0]
            if split == 'train':
                self.ims = self.ims[:int(N * 0.8)]
                self.labels = self.labels[:int(N * 0.8)]
            elif split == 'val':
                self.ims = self.ims[int(N * 0.8):]
                self.labels = self.labels[int(N * 0.8):]
            else:
                raise NotImplementedError

        attr = self._load_csv("list_attr_celeba.txt", header=1)
        self.attr_names = [x.lower() for x in attr.header if x]
        print(self.attr_names)

        # select certain attribute
        if attr_name is None:
            self.all = False
            print('training unconditional celeba model')
        elif attr_name == 'all':
            self.all = True
            self.attr_name = attr_name
            self.attr_names_idx = {self.attr_names[i].lower(): i for i in range(len(self.attr_names))}
            print(f'labels are including all attributes')
        else:
            self.all = False
            self.attr_name = attr_name
            self.attr_names_idx = {self.attr_names[i].lower(): i for i in range(len(self.attr_names))}
            self.attr_name_idx = self.attr_names_idx[verify_str_arg(attr_name.lower(), "attr_name", self.attr_names)]
            self.labels = self.labels[:, self.attr_name_idx]
            print(f'labels are related to the attribute: {attr_name}')

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.ims[index]).convert('RGB')
        attr = self.labels[index]
        label = 1

        if random.uniform(0, 1) < 0.5 and self.split == 'train':
            if random.uniform(0, 1) < 0.5:
                # sample negative relation
                attr = 0 if attr == 1 else 1
                label = 0
            else:
                # sample negative image
                neg_idx = random.randint(0, len(self.ims) - 1)
                while neg_idx == index or self.labels[neg_idx] == self.labels[index]:
                    neg_idx = random.randint(0, len(self.ims) - 1)
                image = Image.fromarray(self.ims[neg_idx]).convert('RGB')
                label = 0

        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        # range 0 to 1
        arr = arr.astype(np.float32) / 255.
        return np.transpose(arr, [2, 0, 1]), attr, label

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))


if __name__ == '__main__':
    dataset = ClevrRelDataset(resolution=128, split='train')
