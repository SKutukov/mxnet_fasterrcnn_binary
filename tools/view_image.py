import cv2
import random
import os
from shutil import copyfile
import argparse
import logging
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='View images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--full_precision_dir', type=str, required=True)
    parser.add_argument('--binary_precision_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=100)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    dir1 = args.full_precision_dir
    dir2 = args.binary_precision_dir
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        logger.info('{} already exist!!!'.format(out_dir))

    for file in random.sample(os.listdir(dir1), args.sample_size):
        img_fp = cv2.imread(os.path.join(dir1, file))
        img_bp = cv2.imread(os.path.join(dir2, file))



        cv2.imshow('full precision', img_fp)
        cv2.imshow('binary precision', img_bp)

        if cv2.waitKey() == ord('s'):
            print('Save {}'.format(file))
            copyfile(os.path.join(dir1, file), os.path.join(out_dir, file.split('.')[0] + '_fp.jpg'))
            copyfile(os.path.join(dir2, file), os.path.join(out_dir, file.split('.')[0] + '_bp.jpg'))

        elif cv2.waitKey() == ord('q'):
            break


