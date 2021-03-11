import os
import cv2


def file_check(paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileExistsError(f'File: {path} does not exists.')


def readim(path: str, color_mode: str='rgb'):
    """ 
    Read image through path and
    convert it into the grayscale.
    """
    mode = {'rgb': 1, 'grayscale': 0}
    image = cv2.imread(path, mode[color_mode])
    return image
