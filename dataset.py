import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data


class PanoDataset(data.Dataset):
    '''
    @root_dir (str)
        path to root directory where all data and
        ground truth located at
    @cat_list (list of str)
        list of sub-directories under root_dir to find .png
        e.g. ['img', 'line']
        filenames list of all sub-directories should the same
        i.e.
            if there is a 'room.png' in '{root_dir}/img/',
            '{root_dir}/line/room.png' have to exist
    @flip (bool)
        whether to performe random left-right flip
    @rotate (bool)
        whether to performe random horizontal angle rotate
    '''
    def __init__(self, root_dir, cat_list, flip=False, rotate=False):
        self.root_dir = root_dir
        self.cat_list = cat_list
        self.fnames = [
            fname for fname in os.listdir(os.path.join(root_dir, cat_list[0]))]
        self.flip = flip
        self.rotate = rotate

        self._check_dataset()

    def _check_dataset(self):
        for fname in self.fnames:
            for cat in self.cat_list:
                cat_path = os.path.join(self.root_dir, cat, fname)
                assert os.path.isfile(cat_path), '%s not found !!!' % cat_path

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path_list = [
            os.path.join(self.root_dir, cat, self.fnames[idx])
            for cat in self.cat_list]
        npimg_list = [
            np.array(Image.open(path), np.float32) / 255
            for path in path_list]

        # Random flip
        if self.flip and np.random.randint(2) == 0:
            npimg_list = [np.flip(npimg, axis=1) for npimg in npimg_list]

        # Random horizontal rotate
        if self.rotate:
            dx = np.random.randint(npimg_list[0].shape[1])
            npimg_list = [np.roll(npimg, dx, axis=1) for npimg in npimg_list]

        # Transpose to C x H x W
        npimg_list = [
            np.expand_dims(npimg, axis=0) if npimg.ndim == 2 else npimg.transpose([2, 0, 1])
            for npimg in npimg_list]
        return (torch.FloatTensor(npimg) for npimg in npimg_list)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='data/train')
    parser.add_argument('--cat_list', default=['img', 'line', 'edge', 'cor'], nargs='+')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--rotate', action='store_true')
    args = parser.parse_args()

    print('args:')
    for key, val in vars(args).items():
        print('    {:16} {}'.format(key, val))

    dataset = PanoDataset(
        root_dir=args.root_dir, cat_list=args.cat_list,
        flip=args.flip, rotate=args.rotate)
    print('len(dataset): {}'.format(len(dataset)))

    for ith, x in enumerate(dataset[0]):
        print(
            'size', x.size(),
            '| dtype', x.dtype,
            '| mean', x.mean().item(),
            '| std', x.std().item(),
            '| min', x.min().item(),
            '| max', x.max().item())
