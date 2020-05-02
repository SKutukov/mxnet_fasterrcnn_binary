import argparse
import mxnet as mx
from symnet.model import load_param, check_shape
from mxnet.module import Module
from convert.convert_maps import get_weight_map
import ast

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
    parser.add_argument('--rpn-feat-stride', type=int, default=16)
    parser.add_argument('--rpn-anchor-scales', type=str, default='(8, 16, 32)')
    parser.add_argument('--rpn-anchor-ratios', type=str, default='(0.5, 1, 2)')
    parser.add_argument('--rpn-pre-nms-topk', type=int, default=12000)
    parser.add_argument('--rpn-post-nms-topk', type=int, default=2000)
    parser.add_argument('--rpn-nms-thresh', type=float, default=0.7)
    parser.add_argument('--rpn-min-size', type=int, default=16)
    parser.add_argument('--rpn-batch-rois', type=int, default=256)
    parser.add_argument('--rpn-allowed-border', type=int, default=0)
    parser.add_argument('--rpn-fg-fraction', type=float, default=0.5)
    parser.add_argument('--rpn-fg-overlap', type=float, default=0.7)
    parser.add_argument('--rpn-bg-overlap', type=float, default=0.3)
    parser.add_argument('--rcnn-num-classes', type=int, default=21)
    parser.add_argument('--rcnn-feat-stride', type=int, default=16)
    parser.add_argument('--rcnn-pooled-size', type=str, default='(14, 14)')
    parser.add_argument('--rcnn-batch-size', type=int, default=1)
    parser.add_argument('--rcnn-batch-rois', type=int, default=128)
    parser.add_argument('--rcnn-fg-fraction', type=float, default=0.25)
    parser.add_argument('--rcnn-fg-overlap', type=float, default=0.5)
    parser.add_argument('--rcnn-bbox-stds', type=str, default='(0.1, 0.1, 0.2, 0.2)')
    parser.add_argument('--is_bin_old', action='store_true', default=False)
    parser.add_argument('--step_old', type=int, default=1)
    parser.add_argument('--step_new', type=int, default=1)
    parser.add_argument('--is_bin_new', action='store_true', default=False)
    parser.add_argument('--img-pixel-means', type=str, default='(0.0, 0.0, 0.0)')
    parser.add_argument('--img-pixel-stds', type=str, default='(1.0, 1.0, 1.0)')
    parser.add_argument('--net-fixed-params', type=str, default='["conv0", "stage1", "gamma", "beta"]')

    args = parser.parse_args()
    args.img_pixel_means = ast.literal_eval(args.img_pixel_means)
    args.img_pixel_stds = ast.literal_eval(args.img_pixel_stds)
    args.net_fixed_params = ast.literal_eval(args.net_fixed_params)
    args.rpn_anchor_scales = ast.literal_eval(args.rpn_anchor_scales)
    args.rpn_anchor_ratios = ast.literal_eval(args.rpn_anchor_ratios)
    args.rcnn_pooled_size = ast.literal_eval(args.rcnn_pooled_size)
    args.rcnn_bbox_stds = ast.literal_eval(args.rcnn_bbox_stds)
    return args

def get_vgg16_test(args):
    from symnet.symbol_vgg import get_vgg_test
    if not args.params:
        args.params = 'model/vgg16-0010.params'
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (7, 7)
    return get_vgg_test(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                        rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                        rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                        rpn_min_size=args.rpn_min_size,
                        num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                        rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                        isBin=args.is_bin_new,
                        step=args.step_new)


def convert_net(sym, args):
    # setup context
    ctx = mx.cpu(0)

    weight_map = get_weight_map(args.step_old, args.is_bin_old,
                                args.step_new, args.is_bin_new)
    # load params
    arg_params, aux_params = load_param(args.params, ctx=ctx, weight_map=weight_map)

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
    net = get_vgg16_test(args)
    convert_net(net, args)
