import numpy as np
from skimage.util.shape import view_as_windows


def split_image(image, split_size: int,
          stride: int, padding: bool):
    """ Splitting the images into blocks """
    gray = True if len(image.shape) < 3 else False

    # image padding
    if padding:
        h, w = image.shape[:2]
        ph = stride - ((h - split_size) % stride)
        pw = stride - ((w - split_size) % stride)
        pad_size = ((0, ph), (0, pw)) + (() if gray else ((0,0),))
        image = np.pad(image, pad_size)

    # splitting into blocks
    split_shape = (split_size, split_size) + (() if gray else (3,))
    blocks = view_as_windows(image, split_shape, stride)
    blocks = blocks.reshape(-1, *split_shape)

    return blocks
