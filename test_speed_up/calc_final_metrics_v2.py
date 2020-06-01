import os
import numpy as np
import argparse
import logging
import sys
import scipy.stats as st
import random

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def main(args):
    logger.addHandler(logging.FileHandler(args.log_path))

    with open(args.filename_with_or_time) as or_times, \
        open(args.filename_with_bin_time) as bin_times:
        data_or = [float(x) for x in or_times.readlines()]
        data_bin = [float(x) for x in bin_times.readlines()]

        speed_ups = []
        for i in range(0, 100):
            data_or_sample = random.sample(data_or, len(data_bin)//10)
            data_bin_sample = random.sample(data_bin, len(data_bin)//10)
            data_or_sample = np.array(data_or_sample)
            data_bin_sample = np.array(data_bin_sample)
            speed_up = data_or_sample.sum()/data_bin_sample.sum()
            speed_ups.append(speed_up)
        # data = np.array(data)
        interval = st.t.interval(0.95, len(speed_ups) - 1, loc=np.mean(speed_ups), scale=st.sem(speed_ups))
        print(round((interval[0] + interval[1]) / 2, 2), round((interval[1] - interval[0]) / 2, 3))

        logger.info("Speed UP Mean: {:3f}, STD: {:3f}".format(round((interval[0] + interval[1]) / 2, 2),
                                                              round((interval[1] - interval[0]) / 2, 3)))

    or_file_size = os.stat(args.or_param_path).st_size
    binarized_file_size = os.stat(args.binarized_param_path).st_size
    comp = (1 - binarized_file_size / or_file_size) * 100
    logger.info("Weight comprasion {:3f} %".format(comp))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--filename_with_or_time", type=str)
    parser.add_argument("--filename_with_bin_time", type=str)
    parser.add_argument("--or_param_path", type=str)
    parser.add_argument("--binarized_param_path", type=str)
    parser.add_argument("--log_path", type=str, default='result.txt')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args=args)
