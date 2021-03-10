import argparse
from pkgs.pascal import PascalVOCDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Split Tools")
    parser.add_argument('--im-path', type=str, default=None,
                        help='input image path')
    parser.add_argument('--lb-path', type=str, default=None,
                        help='segmentation label image path')
    parser.add_argument('--split-size', type=int, default=224,
                        help='block size (default: 224)')
    parser.add_argument('--padding', action='store_true', default=False,
                        help='whether padding the image to match the split size (default: false)')
    parser.add_argument('--stride', type=int, default=None,
                        help='sliding window slide (default: auto)')
    parser.add_argument('--output-format', type=str, default='.png',
                        choices=['.png', '.jpg'], help='output images format (default: .png)')
    parser.add_argument('--color-mode', type=str, default='rgb',
                        choices=['rgb', 'grayscale'], help='image reading mode (default: rgb)')
    parser.add_argument('--train-val-split', action='store_true', default=False,
                        help='whether to generate the train and val split data.')
    parser.add_argument('--split-ratio', type=float, default=0.05,
                        help='splitting ratio of train and val dataset')
    parser.add_argument('--background-filter', action='store_true', default=False,
                        help='filtering the patches with specific background ratio')
    parser.add_argument('--background-ratio', type=float, default=0.95, 
                        help='upper bound of background ratio in each patch')
    args = parser.parse_args()

    if args.stride is None or args.stride >= args.split_size:
        args.stride = args.split_size

    if not args.train_val_split:
        args.split_ratio = 0.
    elif args.split_ratio <= 0. or args.split_ratio >= 1.:
        raise ValueError('Split ratio must be in the range (0, 1).')

    if args.background_filter:
        assert 0 <= args.background_ratio <= 1, 'Background ratio must be in the range [0, 1].'

    dataset = PascalVOCDataset('PascalVOC', args)
