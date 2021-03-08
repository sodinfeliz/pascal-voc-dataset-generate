import os
import shutil
import argparse
import cv2
import numpy as np
from skimage.util.shape import view_as_windows


mode = {
    'origin': -1,
    'rgb': 1,
    'grayscale': 0
}


def split(args):
    image = cv2.imread(args.im_path, mode[args.color_mode])
    if args.color_mode == 'grayscale':
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = image[..., :3]
    split_shape = (args.split_size, args.split_size, 3)

    # image padding
    if args.padding:
        h, w = image.shape[:2]
        ph = (args.split_size - h % args.split_size) % args.split_size
        pw = (args.split_size - w % args.split_size) % args.split_size
        image = np.pad(image, ((0, ph), (0, pw), (0, 0)))

    # splitting into blocks
    blocks = view_as_windows(image, split_shape, args.stride)
    blocks = blocks.reshape(-1, *split_shape)

    folder = './splits'
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

    for idx, block in enumerate(blocks):
        cv2.imwrite(os.path.join(folder, f'{idx}{args.output_format}'), block)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Split Tools")
    parser.add_argument('--im-path', type=str, default=None,
                        help='input image path')
    parser.add_argument('--split-size', type=int, default=224,
                        help='block size (default: 224)')
    parser.add_argument('--padding', action='store_true', default=False,
                        help='whether padding the image to match the split size (default: false)')
    parser.add_argument('--stride', type=int, default=None,
                        help='sliding window slide (default: auto)')
    parser.add_argument('--output-format', type=str, default='.png',
                        choices=['.png', '.jpg'], help='output images format (default: .png)')
    parser.add_argument('--color-mode', type=str, default='origin',
                        choices=['origin', 'rgb', 'grayscale'],
                        help='reading image mode (default: origin)')
    args = parser.parse_args()

    if args.im_path is None:
        raise ValueError('Must specified the input image path')
    elif not os.path.exists(args.im_path):
        raise FileExistsError(f'File: {args.im_path} does not exists.')

    if args.stride is None or args.stride >= args.split_size:
        args.stride = args.split_size // 2

    split(args)


