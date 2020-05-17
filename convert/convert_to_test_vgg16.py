import argparse
import mxnet as mx
from symnet.model import load_param, check_shape
from mxnet.module import Module
from convert.convert_maps import get_weight_map
import ast
from symnet.factory import get_network
from tools.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Comvert Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='vgg16', help='base network')
    parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--params', type=str, default='', help='path to last saved model')
    parser.add_argument('--save-prefix', type=str, default='', help='path to last saved model')
    # faster rcnn params
    parser.add_argument('--img-short-side', type=int, default=600)
    parser.add_argument('--img-long-side', type=int, default=1000)
    parser.add_argument('--rpn-batch-rois', type=int, default=256)
    parser.add_argument('--rpn-allowed-border', type=int, default=0)
    parser.add_argument('--rcnn-batch-size', type=int, default=1)
    parser.add_argument('--rcnn-batch-rois', type=int, default=128)
    parser.add_argument('--is_bin', action='store_true', default=False)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--config_filename', type=str)
    args = parser.parse_args()
    return args


def convert_net(sym, args):
    # setup context
    ctx = mx.cpu(0)

    # weight_map = get_weight_map(args.step_old, args.is_bin_old,
    #                             args.step_new, args.is_bin_new)
    # load params
    arg_params, aux_params = load_param(args.params, ctx=ctx)

    # produce shape max possible
    data_names = ['data', 'im_info']
    label_names = None
    data_shapes = [('data', (1, 3, args.img_long_side, args.img_long_side)), ('im_info', (1, 3))]
    label_shapes = None

    # check shapes
    check_shape(sym, data_shapes, arg_params, aux_params)

    # create and bind module
    mod = Module(sym, data_names, label_names, context=ctx)
    mod.bind(data_shapes, label_shapes, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    # forward
    mod.save_checkpoint(args.save_prefix, epoch=0)


if __name__ == '__main__':
    args = parse_args()
    config = Config(args.config_filename)
    net = get_network(args.network, args, config, 'test')
    convert_net(net, args)
