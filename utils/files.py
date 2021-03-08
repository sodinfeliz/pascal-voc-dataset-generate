import os


def file_check(path):
    if path is None:
        raise ValueError('Must specified the input image path')
    elif not os.path.exists(path):
        raise FileExistsError(f'File: {path} does not exists.')

