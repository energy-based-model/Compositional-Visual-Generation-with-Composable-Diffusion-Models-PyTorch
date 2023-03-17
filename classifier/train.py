# utility functions based off https://github.com/yilundu/improved_contrastive_divergence
# @article{du2020improved,
#   title={Improved Contrastive Divergence Training of Energy Based Models},
#   author={Du, Yilun and Li, Shuang and Tenenbaum, Joshua and Mordatch, Igor},
#   journal={arXiv preprint arXiv:2012.01316},
#   year={2020}
# }
import numpy as np

import os
import time
import copy
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model import ResNetModel

from torch.utils.data import Dataset, DataLoader

from datasets import Clevr2DPosDataset, FFHQSubdataset, ClevrRelDataset, CelebA128dataset

from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as tvu

# seed
np.random.seed(301)
torch.manual_seed(301)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(301)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_binary_classification(**kwargs):
    model = ResNetModel(**kwargs)
    model = model.train().to(device)

    print(kwargs)

    dataset = kwargs.get('dataset', '')
    lr = kwargs.get('lr', 1e-4)
    batch_size = kwargs.get('batch_size', 32)
    checkpoint_dir = kwargs.get('checkpoint_dir', './results')
    cond_idx = kwargs.get('cond_idx', None)
    im_size = kwargs.get('im_size', 64)
    filter_dim = kwargs.get('filter_dim')
    attribute = kwargs.get('attribute', None)


    if dataset == 'ffhq':
        assert kwargs.get('cond_idx') is not None

    if dataset == 'clevr_pos':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        datasets = {phase: Clevr2DPosDataset(resolution=im_size, split=phase) for phase in ['train', 'val']}
    elif dataset == 'ffhq':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        datasets = {phase: FFHQSubdataset(resolution=im_size, split=phase, cond_idx=cond_idx) for phase in ['train', 'val']}
    elif dataset == 'clevr_rel':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        datasets = {phase: ClevrRelDataset(resolution=im_size, split=phase) for phase in ['train', 'val']}
    elif dataset == 'celeba':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        datasets = {phase: CelebA128dataset(root='dataset', hq=True,
                                            attr_name=attribute, resolution=im_size, split=phase)
                    for phase in ['train', 'val']}
    else:
        raise NotImplementedError

    dataset_sizes = {phase: len(datasets[phase]) for phase in ['train', 'val']}
    dataloaders = {phase: DataLoader(
        dataset=datasets[phase], shuffle=True, pin_memory=True, num_workers=4, batch_size=batch_size)
        for phase in ['train', 'val']
    }

    criterion = nn.CrossEntropyLoss()

    if dataset == 'clevr_pos':
        checkpoint_path = os.path.join(checkpoint_dir, f'{dataset}_classifier_{im_size}')
    elif dataset == 'ffhq':
        attr = {0: 'smile', 1: 'glasses', 3: 'gender'}
        checkpoint_path = os.path.join(checkpoint_dir, f'{dataset}_{attr[cond_idx]}_classifier_{im_size}')
    elif dataset == 'clevr_rel':
        checkpoint_path = os.path.join(checkpoint_dir, f'{dataset}_classifier_{im_size}')
    elif dataset == 'celeba':
        checkpoint_path = os.path.join(checkpoint_dir, f'{dataset}_classifier_{attribute}_{im_size}')
    else:
        raise NotImplementedError

    os.makedirs(checkpoint_path, exist_ok=True)
    log_dir = checkpoint_path + '/logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(str(log_dir))
    train_model(model, dataloaders, criterion, optimizer, dataset_sizes, checkpoint_path, 0, 1000, dataset, logger)


def train_model(
        model, dataloaders, criterion, optimizer, dataset_sizes,
        checkpoint_path, start_epoch=0, num_epochs=50, dataset=None, logger=None
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_index = 0

            # Iterate over data.
            for i, (inputs, attr, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.float().to(device)
                labels = labels.float()[..., None].to(device)

                if dataset == 'clevr_pos':
                    attr = attr.float().to(device)
                elif dataset == 'ffhq':
                    attr = attr.to(device)
                elif dataset == 'clevr_rel':
                    attr = attr.to(device)
                elif dataset == 'celeba':
                    attr = attr.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'val':
                        model.eval()
                    else:
                        model.train()

                    logits = model(inputs, attr)

                    loss = criterion(logits, labels[:,0].long())
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    preds = torch.zeros([logits.shape[0]]).to(device)
                    preds[logits[:, 0] < logits[:, 1]] = 1
                    corrects = torch.sum(preds == labels[:, 0]) / labels.shape[0]

                    running_loss += loss.item()
                    running_corrects += corrects.item()
                    running_index += 1

            epoch_loss = running_loss / running_index
            epoch_acc = running_corrects / running_index
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                logger.add_scalar('val/loss', epoch_loss, epoch)
                logger.add_scalar('val/acc', epoch_acc, epoch)
            else:
                logger.add_scalar('train/loss', epoch_loss, epoch)
                logger.add_scalar('train/acc', epoch_acc, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'val': best_acc,
                },
                    os.path.join(checkpoint_path, f'{epoch + 1}.tar')
                )

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # models
    parser.add_argument("--spec_norm", action="store_true", default=True)
    parser.add_argument("--norm", action="store_true", default=True)
    parser.add_argument("--alias", action="store_true", default=False)
    parser.add_argument("--filter_dim", type=int, default=64)
    parser.add_argument("--im_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", choices=['clevr_pos', 'ffhq', 'clevr_rel', 'celeba'])
    parser.add_argument("--cond_idx", type=int, choices=[0, 1, 3],
                        help='conditioning for ffhq: 0=smile, 1=glasses, 3=gender')
    parser.add_argument("--attribute", type=str)
    parser.add_argument("--checkpoint_dir", type=str)

    args = parser.parse_args()

    kwargs = dict(
        spec_norm=args.spec_norm,
        norm=args.norm,
        dataset=args.dataset,
        lr=args.lr,
        filter_dim=args.filter_dim,
        im_size=args.im_size,
        cond_idx=args.cond_idx,
        checkpoint_dir=args.checkpoint_dir,
        attribute=args.attribute
    )

    train_binary_classification(**kwargs)
