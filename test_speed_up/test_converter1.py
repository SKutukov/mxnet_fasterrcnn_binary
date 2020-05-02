import pytest
import mxnet as mx
import subprocess
from os.path import abspath

from mxnet import gluon
from mxnet.gluon import nn
from numpy.testing import assert_almost_equal
import argparse
import time
from symdata.loader import load_test, generate_batch

path_to_image = '/home/skutukov/Downloads/Send-Archive/photo_2020-05-01_21-50-35.jpg'
img_short_side = 600
img_long_side = 1000
img_pixel_means = (123.68, 116.779, 103.939)
img_pixel_stds = (1.0, 1.0, 1.0)

def dummy_data(ctx):
    im_tensor, im_info, im_orig = load_test(path_to_image, short=img_short_side, max_size=img_long_side,
                                            mean=img_pixel_means, std=img_pixel_stds)
    data_batch = generate_batch(im_tensor, im_info)
    return mx.nd.random.uniform(shape=(1, 3, 1000, 600), ctx=ctx)


def save(ctx, args):
    test_count = 1000

    # Training loop would be here
    RES_FILENAME = "res_%s.txt" % args.block_part
    if args.part == 1:
        with open(RES_FILENAME, 'w') as res_file:
            # Export as symbol, so it can be used with C API
            orig_net = mx.gluon.nn.SymbolBlock.imports(
                        "/home/skutukov/work/test_model/test-symbol.json",
                        ['data', 'im_info'],
                        param_file="/home/skutukov/work/test_model/test-0000.params",
                        ctx=ctx)

            # This dummy pass is needed to make correct symbol export possible, but does not replace the first one
            data = dummy_data(ctx)
            _ = orig_net(data)
            # orig_net.summary(data)
            # Intermediate symbolic model, non-compressed
            star_time = time.time()
            for i in range(0, test_count):
                _ = orig_net(data)

            dt1 = (time.time() - star_time)/test_count

            print("time original : %f" % dt1)
            res_file.write("time original: %f\n" % dt1)

    # output = subprocess.check_output(["/home/skutukov/work/BMXNet-v2/build/tools/binary_converter/model-converter", param_file])

    with open(RES_FILENAME, 'a') as res_file:

        if args.part == 2:

            with open(RES_FILENAME, 'r') as res_file1:
                line = res_file1.readline()
                print("line", line)
                dt1 = float(line.split('\n')[0].split(' ')[-1])


            # orig_net.rpn.init_feature_extractor_fast(ctx)

            feature_extractor_fast = []
            for block_part in range(0, 8):
                feature_extractor_fast.append(
                    mx.gluon.nn.SymbolBlock.imports(
                        "%s_test-symbol.json" % block_part,
                        ['data'],
                        param_file="%s_test-0000.params" % block_part,
                        ctx=ctx))


            # orig_net.rpn.feature_extractor_fast(data)
            # # Compressed symbolic model
            # net2 = mx.gluon.nn.SymbolBlock.imports(prefix + symbol_file, ['data'], param_file=prefix + param_file, ctx=ctx)
            star_time = time.time()
            for i in range(0, test_count):
                data, _ = dummy_data(ctx)
                for index, block in enumerate(feature_extractor_fast):
                    # data, _ = globals()["dummy_data%s" % i](ctx)
                    data = block(data)

            dt2 = (time.time() - star_time) / test_count

            print("time final compressed : %f" % dt2)
            print("speed up : %f" % (float(dt1) / dt2))
            res_file.write("time_final_compressed: %f\n" % dt2)
            res_file.write("speed_up: %f\n" % (float(dt1) / dt2))


    # assert_almost_equal(expected.asnumpy(), out1.asnumpy())
    # assert_almost_equal(expected.asnumpy(), out2.asnumpy())

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--part", type=int)
    parser.add_argument("--block_part", type=str)

    return parser.parse_args()

if __name__== "__main__":
    args = parse_args()
    save(ctx=mx.cpu(), args=args)
