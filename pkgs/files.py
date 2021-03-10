import os

def file_check(paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileExistsError(f'File: {path} does not exists.')
