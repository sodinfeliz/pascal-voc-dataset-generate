import argparse
from utils.pascal import pascal_voc_generate


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
    parser.add_argument('--color-mode', type=str, default='origin',
                        choices=['origin', 'rgb', 'grayscale'],
                        help='reading image mode (default: origin)')
    args = parser.parse_args()

    if args.stride is None or args.stride >= args.split_size:
        args.stride = args.split_size

    pascal_voc_generate(args)
