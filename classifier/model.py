import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Dropout
from torch.nn.utils import spectral_norm

import numpy as np
import pdb

def swish(x):
    return x * torch.sigmoid(x)


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        if (self.filt_size == 1):
            a = np.array([1., ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Downsample1D(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if (self.filt_size == 1):
            a = np.array([1., ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer_1d(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class CondResBlock(nn.Module):
    def __init__(self, alias=False, downsample=True, rescale=True, filters=64, latent_dim=64, im_size=64, classes=512,
                 norm=True, spec_norm=False, no_res=False):
        super(CondResBlock, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.downsample = downsample
        self.alias = alias
        self.no_res = no_res

        if filters <= 128:
            # self.bn1 = nn.InstanceNorm2d(filters, affine=True)
            self.bn1 = nn.GroupNorm(int(filters / 128 * 32), filters, affine=True)
        else:
            self.bn1 = nn.GroupNorm(32, filters)

        if not norm:
            self.bn1 = None

        if spec_norm:
            self.conv1 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
        else:
            self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        if filters <= 128:
            self.bn2 = nn.GroupNorm(int(filters / 128 * 32), filters, affine=True)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=True)

        if not norm:
            self.bn2 = None

        if spec_norm:
            self.conv2 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
        else:
            self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        self.dropout = Dropout(0.2)

        # Upscale to an mask of image
        self.latent_map = nn.Linear(classes, 2 * filters)
        self.latent_map_2 = nn.Linear(classes, 2 * filters)

        self.relu = torch.nn.ReLU(inplace=True)
        self.act = swish

        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)

                if self.alias:
                    self.avg_pool = Downsample(channels=2 * filters)
                else:
                    self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

                if self.alias:
                    self.avg_pool = Downsample(channels=filters)
                else:
                    self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x, y):
        x_orig = x

        if y is not None:
            latent_map = self.latent_map(y).view(-1, 2 * self.filters, 1, 1)

            gain = latent_map[:, :self.filters]
            bias = latent_map[:, self.filters:]

        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)

        if y is not None:
            x = gain * x + bias

        x = self.act(x)
        # x = self.dropout(x)

        if y is not None:
            latent_map = self.latent_map_2(y).view(-1, 2 * self.filters, 1, 1)
            gain = latent_map[:, :self.filters]
            bias = latent_map[:, self.filters:]

        x = self.conv2(x)

        if self.bn2 is not None:
            x = self.bn2(x)

        # x = self.dropout(x)

        if y is not None:
            x = gain * x + bias

        x = self.act(x)
        # x_out = x

        if not self.no_res:
            x_out = x_orig + x
        else:
            x_out = x

        if self.downsample:
            x_out = self.conv_downsample(x_out)
            x_out = self.act(self.avg_pool(x_out))

        return x_out


class ResNetModel(nn.Module):
    def __init__(self, **kwargs):
        super(ResNetModel, self).__init__()
        self.act = swish
        self.spec_norm = kwargs.get('spec_norm', True)
        self.norm = kwargs.get('norm', True)
        self.dataset = kwargs.get('dataset', '')
        self.filter_dim = kwargs.get('filter_dim', 64)
        self.latent_dim = self.filter_dim
        self.im_size = kwargs.get('im_size', 128)

        if self.dataset == 'clevr_pos':
            self.fc = nn.Linear(2, 16)
            self.input_channel = 3
            self.in_features = 16
        elif self.dataset == 'ffhq' or self.dataset == 'celeba':
            self.input_channel = 3
            self.emb = nn.Embedding(2, 16)
            self.in_features = 16
        elif self.dataset == 'clevr_rel':
            self.input_channel = 3
            self.index_embedding = nn.Embedding(3, 64)
            self.shape_embedding = nn.Embedding(4, 64)
            self.color_embedding = nn.Embedding(9, 64)
            self.size_embedding = nn.Embedding(3, 64)
            self.material_embedding = nn.Embedding(3, 64)
            self.relation_embedding = nn.Embedding(7, 512)
            self.fc = nn.Linear(320, 512)
            self.in_features = 1536
        else:
            raise NotImplementedError

        self.downsample = Downsample(channels=self.input_channel)
        self.init_main_model()

    def embed(self, y):
        if self.dataset == 'clevr_pos':
            return self.fc(y.float())
        elif self.dataset == 'ffhq' or self.dataset == 'celeba':
            return self.emb(y)
        elif self.dataset == 'clevr_rel':
            obj_1 = torch.cat((self.shape_embedding(y[:, 0]), self.size_embedding(y[:, 1]),
                               self.color_embedding(y[:, 2]), self.material_embedding(y[:, 3]),
                               self.index_embedding(y[:, 4])), dim=1)
            obj_2 = torch.cat((self.shape_embedding(y[:, 5]), self.size_embedding(y[:, 6]),
                               self.color_embedding(y[:, 7]), self.material_embedding(y[:, 8]),
                               self.index_embedding(y[:, 9])), dim=1)
            obj_1 = self.fc(obj_1)
            obj_2 = self.fc(obj_2)
            relation_embedding = self.relation_embedding(y[:, 10])
            return torch.cat((obj_1, obj_2, relation_embedding), dim=1)
        else:
            raise NotImplementedError

    def init_main_model(self):
        filter_dim = self.filter_dim
        latent_dim = filter_dim
        im_size = self.im_size
        self.conv1 = nn.Conv2d(self.input_channel, filter_dim, kernel_size=3, stride=1, padding=1)
        self.res_1a = CondResBlock(classes=self.in_features, filters=filter_dim, latent_dim=latent_dim,
                                   im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_1b = CondResBlock(classes=self.in_features, filters=filter_dim, latent_dim=latent_dim,
                                   im_size=im_size, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

        self.res_2a = CondResBlock(classes=self.in_features, filters=filter_dim, latent_dim=latent_dim,
                                   im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_2b = CondResBlock(classes=self.in_features, filters=filter_dim, latent_dim=latent_dim,
                                   im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

        self.res_3a = CondResBlock(classes=self.in_features, filters=2 * filter_dim, latent_dim=latent_dim,
                                   im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_3b = CondResBlock(classes=self.in_features, filters=2 * filter_dim, latent_dim=latent_dim,
                                   im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

        self.res_4a = CondResBlock(classes=self.in_features, filters=4 * filter_dim, latent_dim=latent_dim,
                                   im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
        self.res_4b = CondResBlock(classes=self.in_features, filters=4 * filter_dim, latent_dim=latent_dim,
                                   im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

        self.self_attn = Self_Attn(2 * filter_dim, self.act)

        self.energy_map = nn.Linear(filter_dim * 8, 2)

    def main_model(self, x, latent):
        x = self.act(self.conv1(x))

        x = self.res_1a(x, latent)
        x = self.res_1b(x, latent)

        x = self.res_2a(x, latent)
        x = self.res_2b(x, latent)

        x, _ = self.self_attn(x)

        x = self.res_3a(x, latent)
        x = self.res_3b(x, latent)

        x = self.res_4a(x, latent)
        x = self.res_4b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        energy = self.energy_map(x)

        # pdb.set_trace()
        # energy = torch.sigmoid(energy)
        return energy

    def forward(self, x, y):
        energy = self.main_model(x, self.embed(y))
        return energy
