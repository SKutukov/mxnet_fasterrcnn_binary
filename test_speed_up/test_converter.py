import pytest
import mxnet as mx
from mxnet import autograd
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
    return [im_tensor, im_info]


def save(ctx, args):
    test_count = 200

    # Training loop would be here
    RES_FILENAME = "log.txt"
    RES1_FILENAME = 'time_or.txt'
    RES2_FILENAME = 'time_bn.txt'
    SPEEDUP_FILENAME = "speed.csv"
    if args.part == 1:
        with open(RES_FILENAME, 'w') as res_file, \
             open(RES1_FILENAME, 'a') as res1_file:

            data = mx.symbol.Variable(name="data")
            im_info = mx.symbol.Variable(name="im_info")
            group = mx.symbol.Group([data, im_info])
            # Export as symbol, so it can be used with C API
            orig_net = mx.gluon.nn.SymbolBlock.imports(
                        args.or_symbol_file,
                        ['data', 'im_info'],
                        param_file=args.or_param_file,
                        ctx=ctx)

            data_batch = dummy_data(mx.cpu())
            # orig_net.summary(data)
            # Intermediate symbolic model, non-compressed
            star_time = time.time()
            for i in range(0, test_count):
                with autograd.predict_mode():
                    _ = orig_net(data_batch[0], data_batch[1])


            dt1 = (time.time() - star_time)/test_count

            print("time original : %f" % dt1)
            res_file.write("time original: %f\n" % dt1)
            res1_file.write("%f\n" % dt1)

    # output = subprocess.check_output(["/home/skutukov/work/BMXNet-v2/build/tools/binary_converter/model-converter", param_file])

    with open(RES_FILENAME, 'a') as res_file, \
         open(SPEEDUP_FILENAME, 'a') as CSV_file, \
         open(RES2_FILENAME, 'a') as res2_file:


        if args.part == 2:

            with open(RES_FILENAME, 'r') as res_file1:
                line = res_file1.readline()
                print("line", line)
                dt1 = float(line.split('\n')[0].split(' ')[-1])

            orig_net = mx.gluon.nn.SymbolBlock.imports(
                args.bin_symbol_file,
                ['data', 'im_info'],
                param_file=args.bin_param_file,
                ctx=ctx)

            # This dummy pass is needed to make correct symbol export possible, but does not replace the first one
            data_batch = dummy_data(ctx)
            # orig_net.summary(data)
            # Intermediate symbolic model, non-compressed
            star_time = time.time()
            for i in range(0, test_count):
                with autograd.predict_mode():
                    _ = orig_net(data_batch[0], data_batch[1])

            dt2 = (time.time() - star_time) / test_count
            res2_file.write("%f\n" % dt2)
            print("time final compressed : %f" % dt2)
            print("speed up : %f" % (float(dt1) / dt2))
            res_file.write("time_final_compressed: %f\n" % dt2)
            res_file.write("speed_up: %f\n" % (float(dt1) / dt2))
            CSV_file.write(str((float(dt1) / dt2)) + '\n')


    # assert_almost_equal(expected.asnumpy(), out1.asnumpy())
    # assert_almost_equal(expected.asnumpy(), out2.asnumpy())

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--part", type=int)
    parser.add_argument("--or_param_file", type=str)
    parser.add_argument("--or_symbol_file", type=str)
    parser.add_argument("--bin_param_file", type=str)
    parser.add_argument("--bin_symbol_file", type=str)

    return parser.parse_args()

if __name__== "__main__":
    args = parse_args()
    save(ctx=mx.cpu(), args=args)
