# Adapted from https://github.com/pytorch/vision/blob/v0.2.0/torchvision/datasets/svhn.py
# The difference is here train and extra are mixed.

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import numpy as np
from torchvision.datasets.utils import download_url, check_integrity


class SVHNMIX(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373"
        ],
        'test':
        ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat", "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7"
        ],
        'train_extra': [
            [
                "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
            ],
            ["train_32x32.mat", "extra_32x32.mat"],
            ["e26dedcc434d2e4c54c9b2d4a06d8373", "a93ce644f1a588dc4d68dda5feec44a7"],
        ],
        'train_extra_auto_quan': [
            [
                "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
            ],
            ["train_32x32.mat", "extra_32x32.mat"],
            ["e26dedcc434d2e4c54c9b2d4a06d8373", "a93ce644f1a588dc4d68dda5feec44a7"],
        ],
        'val_auto_quan': [
            [
                "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
            ],
            ["train_32x32.mat", "extra_32x32.mat"],
            ["e26dedcc434d2e4c54c9b2d4a06d8373", "a93ce644f1a588dc4d68dda5feec44a7"],
        ],
    }

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" ' 'or split="extra" or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        if 'train_extra' not in self.split and 'auto_quan' not in self.split:
            # reading(loading) mat file as array
            loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

            self.data = loaded_mat['X']
            # loading from the .mat file gives an np array of type np.uint8
            # converting to np.int64, so that we have a LongTensor after
            # the conversion from the numpy array
            # the squeeze is needed to obtain a 1D tensor
            self.labels = loaded_mat['y'].astype(np.int64).squeeze()

            # the svhn dataset assigns the class label "10" to the digit 0
            # this makes it inconsistent with several loss functions
            # which expect the class labels to be in the range [0, C-1]
            np.place(self.labels, self.labels == 10, 0)
            self.data = np.transpose(self.data, (3, 2, 0, 1))
        else:
            data = []
            labels = []

            for filename in self.filename:
                loaded_mat = sio.loadmat(os.path.join(self.root, filename))
                data_i = loaded_mat['X']
                labels_i = loaded_mat['y'].astype(np.int64).squeeze()
                np.place(labels_i, labels_i == 10, 0)
                data_i = np.transpose(data_i, (3, 2, 0, 1))

                data.append(data_i)
                labels.append(labels_i)

            self.data = np.concatenate(data, axis=0)
            self.labels = np.concatenate(labels, axis=0)

            if self.split == 'train_extra_auto_quan':
                random.seed(123)
                num_train = int(self.data.shape[0] * 0.95)
                all_idx = list(range(self.data.shape[0]))
                random.shuffle(all_idx)
                train_idx = all_idx[0:num_train]
                self.data = np.take(self.data, train_idx, axis=0)
                self.labels = np.take(self.labels, train_idx, axis=0)
            elif self.split == 'val_auto_quan':
                random.seed(123)
                num_train = int(self.data.shape[0] * 0.95)
                all_idx = list(range(self.data.shape[0]))
                random.shuffle(all_idx)
                val_idx = all_idx[num_train:]
                self.data = np.take(self.data, val_idx, axis=0)
                self.labels = np.take(self.labels, val_idx, axis=0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        if 'train_extra' in self.split or 'auto_quan' in self.split:
            for md5, filename in zip(self.file_md5, self.filename):
                fpath = os.path.join(self.root, filename)
                if not check_integrity(fpath, md5):
                    return False
            return True
        else:
            root = self.root
            md5 = self.split_list[self.split][2]
            fpath = os.path.join(root, self.filename)
            return check_integrity(fpath, md5)

    def download(self):
        if 'train_extra' in self.split or 'auto_quan' in self.split:
            for md5, url, filename in zip(self.file_md5, self.url, self.filename):
                download_url(url, self.root, filename, md5)
        else:
            md5 = self.split_list[self.split][2]
            download_url(self.url, self.root, self.filename, md5)