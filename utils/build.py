# --------------------------------------------------------
# Reference from https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------

import os
from random import shuffle

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms.functional as TF
#from .samplers import SubsetRandomSampler
from torch.distributions import Beta
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', ]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images


def make_dataset_officehome(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], int(val.split()[1]), val.split()[2:]) for val in image_list] #TODO
        #print(images)
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


class ObjectImage(torch.utils.data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader, return_path=False):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.return_path = return_path

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            if type(self.transform).__name__ == 'list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        if not self.return_path:
            return img, target
        else:
            return img, target, path

    def __len__(self):
        return len(self.imgs)


class ObjectImage_mul(torch.utils.data.Dataset):
    """
    :return img, label, index for pseudo labels
    """

    def __init__(self, root, label, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.idx2cls = []
        for p, _ in self.imgs:
            cls = p.split('/')[4]
            if cls not in self.idx2cls:
                self.idx2cls.append(cls)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            if type(self.transform).__name__ == 'list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList_idx(torch.utils.data.Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset_officehome(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("no image"))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.idx2cls = []
        for p, _ in self.imgs:
            cls = p.split('/')[4]
            if cls not in self.idx2cls:
                self.idx2cls.append(cls)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


def data_load(args, config, transf=None): 
    ## office home data
    dsets = {}
    dset_loaders = {}
    train_bs = config.DATA.BATCH_SIZE
    target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
    source_root = os.path.join(config.DATA.DATA_PATH, config.DATA.SOURCE + '.txt')
    txt_tar = open(target_root).readlines()
    src_tar = open(source_root).readlines()

    transform = build_transform(is_train=False, config=config) if transf is None else transf
    dsets["target"] = ImageList_idx(txt_tar, transform=transform)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=config.DATA.NUM_WORKERS, drop_last=False)

    transform = build_transform(is_train=True, config=config) if transf is None else transf
    dsets['source'] = ImageList_idx(src_tar, transform=transform)
    dset_loaders['source'] = DataLoader(dsets['source'], batch_size=train_bs, shuffle=True, num_workers=config.DATA.NUM_WORKERS, drop_last=False)

    transform = build_transform(is_train=False, config=config) if transf is None else transf
    dsets["val"] = ImageList_idx(txt_tar, transform=transform)
    dset_loaders["val"] = DataLoader(dsets["val"], batch_size=train_bs, shuffle=False, num_workers=config.DATA.NUM_WORKERS, drop_last=False)

    return dsets, dset_loaders
 

def build_tar_loader(args, config, def_trans=None):
    dsets = {
        'target_train': {},
        'val': {},
    }
    dset_loaders = {
        'target_train': {},
        'val': {},
    }
    transform = build_transform(is_train=True, config=config) if def_trans is None else def_trans
    target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
    dsets['target_train'] = ObjectImage_mul('', target_root, transform)
    transform = build_transform(is_train=False, config=config) if def_trans is None else def_trans
    dsets['val'] = ObjectImage_mul('', target_root, transform)

    dset_loaders['target_train'] = torch.utils.data.DataLoader(
        dsets['target_train'], shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    dset_loaders['val'] = torch.utils.data.DataLoader(
        dsets['val'], 
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dsets, dset_loaders


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    dataset = config.DATA.DATASET
    bicubic = transforms.InterpolationMode.BICUBIC
    if is_train:
        if dataset != 'visda':
            transform = transforms.Compose([
                transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32), interpolation=bicubic),
                transforms.RandomCrop(config.DATA.IMG_SIZE),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32), interpolation=bicubic),
                transforms.CenterCrop(config.DATA.IMG_SIZE),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    else:
        transform = transforms.Compose([
            transforms.Resize(config.DATA.IMG_SIZE, interpolation=bicubic),
            transforms.CenterCrop(config.DATA.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    return transform


class ResizeImage:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))
