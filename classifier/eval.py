import os
import random
import argparse

import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from model import ResNetModel
from datasets import random_crop_arr, center_crop_arr
import pdb

class ClassificationDataset(Dataset):
    def __init__(
        self,
        npy_path,
        resolution,
        test_img_folder,
        random_crop=False,
        random_flip=False,
    ):
        self.resolution = resolution
        self.val_path = npy_path
        self.random_crop = random_crop
        self.random_flip = random_flip

        EXTS = ['jpg', 'jpeg', 'png']
        self.paths = sorted([p for ext in EXTS for p in Path(f'{test_img_folder}').glob(f'**/*.{ext}')])

        # load data
        data = np.load(self.val_path)
        self.labels = data['coords_labels']
        print(f'evaluating {len(self.paths)} images...')

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        gt_label = self.labels[index]

        if self.random_crop:
            arr = random_crop_arr(img, self.resolution)
        else:
            arr = center_crop_arr(img, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        # range 0 to 1
        arr = arr.astype(np.float32) / 255.
        return np.transpose(arr, [2, 0, 1]), gt_label

    def __len__(self):
        return len(self.paths)


class ClassificationFFHQDataset(Dataset):
    def __init__(
        self,
        npy_path,
        resolution,
        test_img_folder,
        random_crop=False,
        random_flip=False,
    ):
        self.resolution = resolution
        self.val_path = npy_path
        self.random_crop = random_crop
        self.random_flip = random_flip

        EXTS = ['jpg', 'jpeg', 'png']
        self.paths = sorted([p for ext in EXTS for p in Path(f'{test_img_folder}').glob(f'**/*.{ext}')])

        # load data
        data = np.load(self.val_path)
        self.labels = data['labels']

        if self.labels.shape[1] == 3:
            male_idx = self.labels[:, 2] == 0
            female_idx = self.labels[:, 2] == 1
            self.labels[:, 2][male_idx] = 1
            self.labels[:, 2][female_idx] = 0

        print(f'evaluating {len(self.paths)} images...')

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        gt_label = self.labels[index]

        if self.random_crop:
            arr = random_crop_arr(img, self.resolution)
        else:
            arr = center_crop_arr(img, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        # range 0 to 1
        arr = arr.astype(np.float32) / 255.
        return np.transpose(arr, [2, 0, 1]), gt_label

    def __len__(self):
        return len(self.paths)


class CLEVRRelDataset(Dataset):
    def __init__(
        self,
        resolution,
        npy_path,
        test_img_folder,
        random_crop=False,
        random_flip=False,
    ):
        self.val_path = npy_path
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip

        EXTS = ['jpg', 'jpeg', 'png']
        self.paths = sorted([p for ext in EXTS for p in Path(f'{test_img_folder}').glob(f'**/*.{ext}')])
        # load data
        data = np.load(self.val_path)
        self.labels = data['labels']
        self.size = min(self.labels.shape[0], len(self.paths))
        print(f'evaluating {len(self.paths)} images...')

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        gt_label = self.labels[index]

        if self.random_crop:
            arr = random_crop_arr(img, self.resolution)
        else:
            arr = center_crop_arr(img, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        # range 0 to 1
        arr = arr.astype(np.float32) / 255.
        return np.transpose(arr, [2, 0, 1]), gt_label


def load_classifier(checkpoint_dir, dataset, im_size, filter_dim, attr=None):
    if dataset == 'ffhq':
        folder = os.path.join(checkpoint_dir, f'{dataset}_{attr}_classifier_{im_size}')
    elif dataset == 'clevr_pos':
        folder = os.path.join(checkpoint_dir, f'{dataset}_classifier_{im_size}')
    elif dataset == 'clevr_rel':
        folder = os.path.join(checkpoint_dir, f'{dataset}_classifier_{im_size}')
    elif dataset == 'celeba':
        folder = os.path.join(checkpoint_dir, f'{dataset}_classifier_{attr}_{im_size}')
    else:
        raise NotImplementedError

    try:
        paths = sorted([
            int(str(p).split('/')[-1].replace('.tar', ''))
            for ext in ['tar'] for p in Path(f'{folder}').glob(f'*.{ext}')
        ])
    except e:
        raise ValueError(f"No models are found in {folder}")

    latest_checkpoint_path = os.path.join(folder, f'{paths[-1]}.tar')
    checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
    print(f'loading from the latest checkpoint: {latest_checkpoint_path}')

    kwargs = dict(
        spec_norm=True,
        norm=True,
        dataset=dataset,
        lr=1e-5,
        filter_dim=filter_dim,
        im_size=im_size
    )

    model = ResNetModel(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model.eval()


def compute_classification_score(classifier, npy_path, generated_img_folder, image_size, mode):
    dataset = ClassificationDataset(npy_path=npy_path, resolution=image_size, test_img_folder=generated_img_folder)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=32, drop_last=False, num_workers=4, pin_memory=True)

    total_corrects, total_ims = 0, 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for i, (gen_ims, gt_labels) in enumerate(tqdm(dataloader)):
            gen_ims = gen_ims.float().to(device)
            if dataset == 'clevr_pos':
                gt_labels = gt_labels.float().to(device)
            else:
                gt_labels = gt_labels.to(device)

            if len(gt_labels.shape) == 3:
                labels = torch.chunk(gt_labels, chunks=gt_labels.shape[1], dim=1)
                labels = [chunk.squeeze(dim=1) for chunk in labels]
            else:
                labels = [gt_labels]

            result = torch.zeros((gen_ims.shape[0]), dtype=torch.long, device=device)
            for label in labels:
                with torch.no_grad():
                    outputs = classifier(gen_ims, label)
                    result += (outputs[:,0] < outputs[:,1]).long()

            corrects = torch.sum(result == len(labels))

            total_corrects += corrects.item()
            total_ims += gen_ims.shape[0]

        print(f'{generated_img_folder} has a classification scores: ', total_corrects / total_ims)


def compute_ffhq_classification_score(classifiers, npy_path, generated_img_folder, image_size, mode):
    dataset = ClassificationFFHQDataset(npy_path=npy_path, resolution=image_size, test_img_folder=generated_img_folder)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=32, drop_last=False, num_workers=4, pin_memory=True)

    total_corrects, total_ims = 0, 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for i, (gen_ims, gt_labels) in enumerate(tqdm(dataloader)):
            gen_ims = gen_ims.float().to(device)
            if dataset == 'clevr_pos':
                gt_labels = gt_labels.float().to(device)
            else:
                gt_labels = gt_labels.to(device)

            result = torch.zeros((gen_ims.shape[0]), dtype=torch.long, device=device)
            for j in range(gt_labels.shape[1]):
                with torch.no_grad():
                    outputs = classifiers[j](gen_ims, gt_labels[:, j])
                    result += (outputs[:,0] < outputs[:,1]).long()

            corrects = torch.sum(result == gt_labels.shape[1])

            total_corrects += corrects.item()
            total_ims += gen_ims.shape[0]

        print(f'{generated_img_folder} has a classification scores: ', total_corrects / total_ims)


def compute_clevr_rel_classification_score(classifier, npy_path, generated_img_folder, image_size, mode):
    dataset = CLEVRRelDataset(npy_path=npy_path, resolution=image_size, test_img_folder=generated_img_folder)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=32, drop_last=False, num_workers=4, pin_memory=True)

    total_corrects, total_ims = 0, 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for i, (gen_ims, gt_labels) in enumerate(tqdm(dataloader)):
        gen_ims = gen_ims.float().to(device)
        gt_labels = gt_labels.to(device)

        if len(gt_labels.shape) == 3:
            labels = torch.chunk(gt_labels, chunks=gt_labels.shape[1], dim=1)
            labels = [chunk.squeeze(dim=1) for chunk in labels]
        else:
            labels = [gt_labels]

        result = torch.zeros((gen_ims.shape[0]), dtype=torch.long, device=device)
        for label in labels:
            with torch.no_grad():
                outputs = classifier(gen_ims, label)

                # pdb.set_trace()
                result += (outputs[:,0] < outputs[:,1]).long()
                # result += (outputs >= 0.5).long()

        corrects = torch.sum(result == len(labels))

        total_corrects += corrects.item()
        total_ims += gen_ims.shape[0]

    print(f'{generated_img_folder} has a classification scores: ', total_corrects / total_ims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classifier flag
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", choices=['clevr_pos', 'ffhq', 'clevr_rel', 'celeba'])
    parser.add_argument("--checkpoint_dir", type=str)

    # input images
    parser.add_argument("--im_size", type=int, default=128)
    parser.add_argument("--npy_path", type=str)
    parser.add_argument("--generated_img_folder", type=str)
    parser.add_argument("--num_rels", type=int)
    parser.add_argument("--mode", choices=['generation'])

    # model
    parser.add_argument("--filter_dim", type=int, default=64)

    args = parser.parse_args()

    if args.dataset == 'clevr_pos':
        classifier = load_classifier(checkpoint_dir=args.checkpoint_dir, dataset=args.dataset,
                                     im_size=args.im_size, filter_dim=args.filter_dim)
        compute_classification_score(
            classifier=classifier, npy_path=args.npy_path,
            generated_img_folder=args.generated_img_folder,
            image_size=args.im_size, mode=args.mode
        )
    elif args.dataset == 'ffhq':
        classifiers = []
        for attr in ['smile', 'glasses', 'gender']:
            classifiers.append(
                load_classifier(checkpoint_dir=args.checkpoint_dir, dataset=args.dataset,
                                attr=attr, im_size=args.im_size, filter_dim=args.filter_dim)
            )
        compute_ffhq_classification_score(
            classifiers=classifiers, npy_path=args.npy_path,
            generated_img_folder=args.generated_img_folder,
            image_size=args.im_size, mode=args.mode
        )
    elif args.dataset == 'celeba':
        classifiers = []
        for attr in ['smiling', 'eyeglasses', 'male']:
            classifiers.append(
                load_classifier(checkpoint_dir=args.checkpoint_dir, dataset=args.dataset,
                                attr=attr, im_size=args.im_size, filter_dim=args.filter_dim)
            )
        compute_ffhq_classification_score(
            classifiers=classifiers, npy_path=args.npy_path,
            generated_img_folder=args.generated_img_folder,
            image_size=args.im_size, mode=args.mode
        )
    elif args.dataset == 'clevr_rel':
        classifier = load_classifier(checkpoint_dir=args.checkpoint_dir, dataset=args.dataset,
                                     im_size=args.im_size, filter_dim=args.filter_dim)
        compute_clevr_rel_classification_score(
            classifier=classifier, npy_path=args.npy_path,
            generated_img_folder=args.generated_img_folder,
            image_size=args.im_size, mode=args.mode
        )
    else:
        raise NotImplementedError


