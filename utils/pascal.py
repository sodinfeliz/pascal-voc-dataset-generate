import os
import shutil
import cv2
from pathlib import Path
from .split import split_image
from .files import file_check


class PascalVOCDataset(object):

    def __init__(self, path):
        self.base_dir = Path(path)
        self.im_dir = self.base_dir.joinpath('JPEGImages')
        self.lb_dir = self.base_dir.joinpath('SegmentationClass')

        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        self.im_dir.mkdir(parents=True)
        self.lb_dir.mkdir(parents=True)


    def save_dataset(self, images, dir, extension='.png'):
        assert dir in ['image', 'label']
        dir = self.im_dir if dir == 'image' else self.lb_dir

        for idx, im in enumerate(images):
            cv2.imwrite(str(dir.joinpath(f'{idx}{extension}')), im)


def pascal_voc_generate(args):
    file_check(args.im_path)
    file_check(args.lb_path)
    images = split_image(args.im_path, args.color_mode, args.split_size, args.stride, args.padding)
    labels = split_image(args.lb_path, args.color_mode, args.split_size, args.stride, args.padding)
    assert len(images) == len(labels), "Inconsistent with image and label image sizes."

    dataset = PascalVOCDataset('PascalVOC')
    dataset.save_dataset(images, 'image', args.output_format)
    dataset.save_dataset(labels, 'label', args.output_format)

