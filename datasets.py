import os
from enum import Enum

from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset as DS
import glob
import os
from collections import defaultdict
import random


INTERP = 3


class Dataset(Enum):
    C10 = 1
    C100 = 2
    STL10 = 3
    IN128 = 4
    PLACES205 = 5


def get_encoder_size(dataset):
    if dataset in [Dataset.C10, Dataset.C100]:
        return 32
    if dataset == Dataset.STL10:
        return 64
    if dataset in [Dataset.IN128, Dataset.PLACES205]:
        return 128
    raise RuntimeError("Couldn't get encoder size, unknown dataset: {}".format(dataset))


def get_dataset(dataset_name):
    try:
        return Dataset[dataset_name.upper()]
    except KeyError as e:
        raise KeyError("Unknown dataset '" + dataset_name + "'. Must be one of "
                       + ', '.join([d.name for d in Dataset]))


class RandomTranslateWithReflect:
    '''
    Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image


class TransformsC10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])
        # transform for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class TransformsSTL10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
        # image augmentation functions
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        rand_crop = \
            transforms.RandomResizedCrop(64, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=INTERP)

        self.test_transform = transforms.Compose([
            transforms.Resize(70, interpolation=INTERP),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize
        ])

        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class TransformsImageNet128:
    '''
    ImageNet dataset, for use with 128x128 full image encoder.
    '''
    def __init__(self):
        # image augmentation functions
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = \
            transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=INTERP)
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(146, interpolation=INTERP),
            transforms.CenterCrop(128),
            post_transform
        ])
        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class myIN128_data_loader(DS):
    def __init__(self, dir, split):
        self.split = split
        self.classes, self.class_to_idx = self.find_classes(dir)
        self.samples_list, self.samples_dict = self.make_dataset(dir, IMG_EXTENSIONS)
                 
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(128),
                transforms.RandomHorizontalFlip(),
                post_transform
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(146, interpolation=INTERP),
                transforms.CenterCrop(128),
                post_transform
            ])
            
    def find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def make_dataset(self, dir, extensions=None, is_valid_file=None):
        images_dict = defaultdict(list)
        images_list = []
        dir = os.path.expanduser(dir)
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return self.has_file_allowed_extension(x, extensions)
        for target in sorted(self.class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        target_label = self.class_to_idx[target]
                        item = (path, target_label)
                        images_list.append(item)
                        images_dict[target_label].append(path)
        return images_list, images_dict

    def has_file_allowed_extension(self, filename, extensions):
        """Checks if a file is an allowed extension.

        Args:
            filename (string): path to a file
            extensions (tuple of strings): extensions to consider (lowercase)

        Returns:
            bool: True if the filename ends with one of given extensions
        """
        return filename.lower().endswith(extensions)

    def __getitem__(self, idx):
        path1, target = self.samples_list[idx]
        image1 = self.transform(Image.open(path1).convert('RGB'))
        if self.split == 'train':
            while True:
                path2 = random.choice(self.samples_dict[target])
                if path2 != path1:
                    image2 = self.transform(Image.open(path2).convert('RGB'))
                    break
            return (image1, image2), target
        else:
            return image1, target

    def __len__(self):
        return len(self.samples_list)

class myC10_data_loader(DS):
    def __init__(self, dir, split):
        self.split = split
        self.imgpaths = self.get_imgpaths(dir, split)
        if split == 'train':
            self.sample_dict = self.build_dict()
        
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        
        if split == 'train':
            self.transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                            ])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                            ])
        self.label_dict = {
                    'airplane':0,
                    'automobile':1,
                    'bird':2,
                    'cat':3,
                    'deer':4,
                    'dog':5,
                    'frog':6,
                    'horse':7,
                    'ship':8,
                    'truck':9,
                    }

    def get_imgpaths(self, dir, split):
        paths = glob.glob(os.path.join(dir, split, '**/*.png'), recursive=True)
        return paths
    
    def build_dict(self):
        sample_dict = defaultdict(list)
        for path in self.imgpaths:
            label = self.get_label(path)
            sample_dict[label].append(path)
        return sample_dict
    
    def get_label(self, path):
        return path.split('_')[-1][:-4]

    def __getitem__(self, idx):
        path = self.imgpaths[idx]
        image = self.transform(Image.open(path).convert('RGB'))
        label = self.get_label(path)
        if self.split == 'train':
            while True:
                path2 = random.choice(self.sample_dict[label])
                if path2 != path:
                    image2 = self.transform(Image.open(path2).convert('RGB'))
                    break
            return (image, image2), self.label_dict[label]
        else:
            return image, self.label_dict[label]

    def __len__(self):
        return len(self.imgpaths)

def build_dataset(dataset, batch_size, input_dir=None, labeled_only=False):

    train_dir, val_dir = _get_directories(dataset, input_dir)

    if dataset == Dataset.C10:
        num_classes = 10
        train_dataset = myC10_data_loader('cifar', 'train')
        test_dataset = myC10_data_loader('cifar', 'test')

    elif dataset == Dataset.C100:
        num_classes = 100
        train_transform = TransformsC10()
        test_transform = train_transform.test_transform
        train_dataset = datasets.CIFAR100(root='/tmp/data/',
                                          train=True,
                                          transform=train_transform,
                                          download=True)
        test_dataset = datasets.CIFAR100(root='/tmp/data/',
                                         train=False,
                                         transform=test_transform,
                                         download=True)
    elif dataset == Dataset.STL10:
        num_classes = 10
        train_transform = TransformsSTL10()
        test_transform = train_transform.test_transform
        train_split = 'train' if labeled_only else 'train+unlabeled'
        train_dataset = datasets.STL10(root='/tmp/data/',
                                       split=train_split,
                                       transform=train_transform,
                                       download=True)
        test_dataset = datasets.STL10(root='/tmp/data/',
                                      split='test',
                                      transform=test_transform,
                                      download=True)
    elif dataset == Dataset.IN128:
        num_classes = 1000
        train_dataset = myIN128_data_loader(train_dir, 'train')
        test_dataset = myIN128_data_loader(val_dir, 'test')

    elif dataset == Dataset.PLACES205:
        num_classes = 1000
        train_transform = TransformsImageNet128()
        test_transform = train_transform.test_transform
        train_dataset = datasets.ImageFolder(train_dir, train_transform)
        test_dataset = datasets.ImageFolder(val_dir, test_transform)

    # build pytorch dataloaders for the datasets
    train_loader = \
        torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    num_workers=16)
    test_loader = \
        torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    num_workers=16)

    return train_loader, test_loader, num_classes


def _get_directories(dataset, input_dir):
    if dataset in [Dataset.C10, Dataset.C100, Dataset.STL10]:
        # Pytorch will download those datasets automatically
        return None, None
    if dataset == Dataset.IN128:
        train_dir = os.path.join(input_dir, 'train/')
        val_dir = os.path.join(input_dir, 'val/')
    elif dataset == Dataset.PLACES205:
        train_dir = os.path.join(input_dir, 'places205_256_train/')
        val_dir = os.path.join(input_dir, 'places205_256_val/')
    else:
        raise 'Data directories for dataset ' + dataset + ' are not defined'
    return train_dir, val_dir
