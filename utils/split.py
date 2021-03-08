import cv2
import numpy as np
from skimage.util.shape import view_as_windows


mode = {
    'origin': -1,
    'rgb': 1,
    'grayscale': 0
}


def split_image(path, color_mode: str, split_size: int,
          stride: int, padding: bool):
    """ Splitting the images into blocks """
    image = cv2.imread(path, mode[color_mode])
    if color_mode == 'grayscale' or len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = image[..., :3]
    split_shape = (split_size, split_size, 3)

    # image padding
    if padding:
        h, w = image.shape[:2]
        ph = (split_size - h % split_size) % split_size
        pw = (split_size - w % split_size) % split_size
        image = np.pad(image, ((0, ph), (0, pw), (0, 0)))

    # splitting into blocks
    blocks = view_as_windows(image, split_shape, stride)
    blocks = blocks.reshape(-1, *split_shape)

    return blocks

