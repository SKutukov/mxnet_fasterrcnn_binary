import os
import numpy
import argparse
import logging
import sys

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def main(args):
    logger.addHandler(logging.FileHandler(args.log_path))

    with open(args.filename_with_speed) as speed_file:
        data = [float(x) for x in speed_file.readlines()]
        data = numpy.array(data)
        logger.info("Speed UP Mean: {:3f}, STD: {:3f}".format(data.mean(), data.std()))

    or_file_size = os.stat(args.or_param_path).st_size
    binarized_file_size = os.stat(args.binarized_param_path).st_size
    comp = (1 - binarized_file_size / or_file_size) * 100
    logger.info("Weight comprasion {:3f} %".format(comp))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--filename_with_speed", type=str)
    parser.add_argument("--or_param_path", type=str)
    parser.add_argument("--binarized_param_path", type=str)
    parser.add_argument("--log_path", type=str, default='result.txt')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args=args)
