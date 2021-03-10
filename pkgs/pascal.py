import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
from tqdm import tqdm
from .split import split_image, readim
from .files import file_check


class PascalVOCDataset(object):

    def __init__(self, path, args):

        file_check([args.im_path, args.lb_path])
        self.args = args
        self.base_dir = Path(path)
        self.im_dir = self.base_dir.joinpath('JPEGImages')
        self.lb_dir = self.base_dir.joinpath('SegmentationClass')

        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        self.im_dir.mkdir(parents=True)
        self.lb_dir.mkdir(parents=True)

        if self.args.split_ratio:
            self.imageset_dir = self.base_dir.joinpath('ImageSets', 'Segmentation')
            self.imageset_dir.mkdir(parents=True)

        self._split_image()
        self._back_detect()
        self._save_dataset()

    
    def _split_image(self):
        self.images = split_image(
            self.args.im_path, self.args.color_mode, 
            self.args.split_size, self.args.stride, 
            self.args.padding)
        self.labels = split_image(
            self.args.lb_path, self.args.color_mode, 
            self.args.split_size, self.args.stride, 
            self.args.padding)
        assert len(self.images) == len(self.labels), "Inconsistent with image and label image sizes."
        self.sz = len(self.images)


    def _back_detect(self):
        if self.args.background_filter:
            org_im = readim(self.args.im_path, self.args.color_mode)
            h, w, short = *org_im.shape[:2], 2000
            if h > w:
                im = cv2.resize(org_im, (w*short//h, short), cv2.INTER_NEAREST)
            else:
                im = cv2.resize(org_im, (short, h*short//w), cv2.INTER_NEAREST)
            im_reshape = im.reshape(-1, 3)
            self.back_color = Counter(map(tuple, im_reshape)).most_common(1)[0][0]


    def _back_ratio(self, patch):
        cdiff = np.abs(patch.reshape(-1, 3).astype('int') - self.back_color)
        pixel_count = self.args.split_size**2 - np.count_nonzero(cdiff.sum(axis=1))
        return pixel_count / self.args.split_size**2


    def _save_dataset(self):
        count = 0
        with tqdm(total=self.sz, desc='Saving Data') as pbar:
            for im, lb in zip(self.images, self.labels):
                pbar.update(1)

                if self.args.background_filter and \
                    self._back_ratio(im) > self.args.background_ratio:
                    continue

                cv2.imwrite(str(self.im_dir.joinpath(f'{count}{self.args.output_format}')), im)
                cv2.imwrite(str(self.lb_dir.joinpath(f'{count}{self.args.output_format}')), lb)
                count += 1

        if self.args.split_ratio:
            val = np.random.choice(count, int(count * self.args.split_ratio), replace=False)
            val.sort()
            train = set(range(count)) - set(val)
            with open(str(self.imageset_dir.joinpath('train.txt')), 'w') as file:
                file.writelines('\n'.join(map(str, train)))
            with open(str(self.imageset_dir.joinpath('val.txt')), 'w') as file:
                file.writelines('\n'.join(map(str, val)))
