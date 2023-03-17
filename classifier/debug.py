import os
import argparse
import torch

from pathlib import Path
from model import ResNetModel
from datasets import ClevrRelDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# classifier flag
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dataset", choices=['clevr_pos', 'ffhq', 'clevr_rel'])
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


def load_classifier(checkpoint_dir, dataset, im_size, filter_dim, attr=None):
    if dataset == 'ffhq':
        folder = os.path.join(checkpoint_dir, f'{dataset}_{attr}_classifier_{im_size}')
    elif dataset == 'clevr_pos':
        folder = os.path.join(checkpoint_dir, f'{dataset}_classifier')
    elif dataset == 'clevr_rel':
        folder = os.path.join(checkpoint_dir, f'{dataset}_classifier_{im_size}')
    else:
        raise NotImplementedError

    paths = sorted([
        int(str(p).split('/')[-1].replace('.tar', ''))
        for ext in ['tar'] for p in Path(f'{folder}').glob(f'**/*.{ext}')
    ])
    latest_checkpoint_path = os.path.join(folder, f'{paths[-1]}.tar')
    checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
    print(f'loading from the latest checkpoint: {latest_checkpoint_path} with val acc: {checkpoint["val"]}')
    kwargs = dict(
        spec_norm=True,
        norm=True,
        dataset=dataset,
        lr=1e-4,
        filter_dim=filter_dim,
        im_size=im_size
    )

    model = ResNetModel(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model.eval()


classifier = load_classifier(checkpoint_dir=args.checkpoint_dir, dataset=args.dataset,
                             im_size=args.im_size, filter_dim=args.filter_dim)

dataset = ClevrRelDataset(resolution=args.im_size, split='train')
dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=10)

total_corrects, total_ims = 0, 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for i, (gen_ims, neg_ims, gt_labels) in enumerate(dataloader):
    gen_ims = gen_ims.float().to(device)
    neg_ims = neg_ims.float().to(device)
    gt_labels = gt_labels.to(device)

    if len(gt_labels.shape) == 3:
        labels = torch.chunk(gt_labels, chunks=gt_labels.shape[1], dim=1)
        labels = [chunk.squeeze(dim=1) for chunk in labels]
    else:
        labels = [gt_labels]

    result = torch.zeros((gen_ims.shape[0], 1), dtype=torch.long, device=device)
    for label in labels:
        with torch.no_grad():
            outputs = classifier(gen_ims, label)
            print(outputs)
            print(classifier(neg_ims, label))
            result += torch.round(outputs).long()

    corrects = torch.sum(result == len(labels))
    total_corrects += corrects.item()
    total_ims += gen_ims.shape[0]
    break

print(f'classification scores: ', total_corrects / total_ims)
