import argparse
import ast
import pprint
import logging

import mxnet as mx
from mxnet.module import Module
import os
from symdata.loader import AnchorGenerator, AnchorSampler, AnchorLoader
from symnet.logger import logger
from symnet.model import load_param, infer_data_shape, check_shape, initialize_frcnn, get_fixed_params, initialize_bias
from symnet.metric import RPNAccMetric, RPNLogLossMetric, RPNL1LossMetric, RCNNAccMetric, RCNNLogLossMetric, RCNNL1LossMetric

from datasets.voc import get_voc_train
from datasets.coco import get_coco_train

from symnet.factory import get_network
from tools.config import Config

def train_net(sym, roidb, args, config):
    logger.addHandler(logging.FileHandler("{0}/{1}".format(args.save_prefix, 'train.log')))
    # print config
    logger.info('called with args\n{}'.format(pprint.pformat(vars(args))))
    # setup multi-gpu
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    batch_size = args.rcnn_batch_size * len(ctx)

    # config = Config('configs/vgg_step_{}.yml'.format(args.step))
    # load training data
    feat_sym = sym.get_internals()['rpn_cls_score_output']
    ag = AnchorGenerator(feat_stride=config.rpn['rpn_feat_stride'],
                         anchor_scales=config.rpn['rpn_anchor_scales'], anchor_ratios=config.rpn['rpn_anchor_ratios'])
    asp = AnchorSampler(allowed_border=args.rpn_allowed_border, batch_rois=args.rpn_batch_rois,
                        fg_fraction=config.rpn['rpn_fg_fraction'], fg_overlap=config.rpn['rpn_fg_overlap'],
                        bg_overlap=config.rpn['rpn_bg_overlap'])
    train_data = AnchorLoader(roidb, batch_size, args.img_short_side, args.img_long_side,
                              config.transform['img_pixel_means'],
                              config.transform['img_pixel_stds'], feat_sym, ag, asp, shuffle=True)

    # produce shape max possible
    _, out_shape, _ = feat_sym.infer_shape(data=(1, 3, args.img_long_side, args.img_long_side))
    feat_height, feat_width = out_shape[0][-2:]
    rpn_num_anchors = len(config.rpn['rpn_anchor_scales']) * len(config.rpn['rpn_anchor_ratios'])
    data_names = ['data', 'im_info', 'gt_boxes']
    label_names = ['label', 'bbox_target', 'bbox_weight']
    data_shapes = [('data', (batch_size, 3, args.img_long_side, args.img_long_side)),
                   ('im_info', (batch_size, 3)),
                   ('gt_boxes', (batch_size, 100, 5))]
    label_shapes = [('label', (batch_size, 1, rpn_num_anchors * feat_height, feat_width)),
                    ('bbox_target', (batch_size, 4 * rpn_num_anchors, feat_height, feat_width)),
                    ('bbox_weight', (batch_size, 4 * rpn_num_anchors, feat_height, feat_width))]

    # print shapes
    data_shape_dict, out_shape_dict = infer_data_shape(sym, data_shapes + label_shapes)
    logger.info('max input shape\n%s' % pprint.pformat(data_shape_dict))
    logger.info('max output shape\n%s' % pprint.pformat(out_shape_dict))

    # load and initialize params
    if args.resume:
        arg_params, aux_params = load_param(args.resume)
        # arg_params, aux_params = initialize_bias(sym, data_shapes, arg_params, aux_params)
    else:
        arg_params, aux_params = load_param(args.pretrained)
        arg_params, aux_params = initialize_frcnn(sym, data_shapes, arg_params, aux_params)
        # arg_params, aux_params = initialize_bias(sym, data_shapes, arg_params, aux_params)
    # check parameter shapes
    check_shape(sym, data_shapes + label_shapes, arg_params, aux_params)

    # check fixed params
    fixed_param_names = get_fixed_params(sym, config.train_param['net_fixed_params'])
    logger.info('locking params\n%s' % pprint.pformat(fixed_param_names))

    # metric
    rpn_eval_metric = RPNAccMetric()
    rpn_cls_metric = RPNLogLossMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    eval_metric = RCNNAccMetric()
    cls_metric = RCNNLogLossMetric()
    bbox_metric = RCNNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callback = mx.callback.Speedometer(batch_size, frequent=args.log_interval, auto_reset=False)
    epoch_end_callback = mx.callback.do_checkpoint(args.save_prefix)

    # learning schedule
    base_lr = args.lr
    lr_factor = 0.1
    lr_epoch = [int(epoch) for epoch in args.lr_decay_epoch.split(',')]
    lr_epoch_diff = [epoch - args.start_epoch for epoch in lr_epoch if epoch > args.start_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    logger.info('lr %f lr_epoch_diff %s lr_iters %s' % (lr, lr_epoch_diff, lr_iters))
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    # train
    mod = Module(sym, data_names=data_names, label_names=label_names,
                 logger=logger, context=ctx, work_load_list=None,
                 fixed_param_names=fixed_param_names)
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore='device',
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=args.start_epoch, num_epoch=args.epochs)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='vgg16', help='base network')
    parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--imageset', type=str, default='', help='imageset splits')
    parser.add_argument('--gpus', type=str, default='0', help='gpu devices eg. 0,1')
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
    parser.add_argument('--lr-decay-epoch', type=str, default='7,15', help='epoch to decay lr')
    parser.add_argument('--resume', type=str, default='', help='path to last saved model')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch for resuming')
    parser.add_argument('--log-interval', type=int, default=100, help='logging mini batch interval')
    parser.add_argument('--save-prefix', type=str, default='', help='saving params prefix')
    # faster rcnn params
    parser.add_argument('--img-short-side', type=int, default=600)
    parser.add_argument('--img-long-side', type=int, default=1000)
    parser.add_argument('--rpn-batch-rois', type=int, default=256)
    parser.add_argument('--rpn-allowed-border', type=int, default=0)
    parser.add_argument('--rcnn-batch-size', type=int, default=1)
    parser.add_argument('--rcnn-batch-rois', type=int, default=128)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--config_filename', type=str)

    args = parser.parse_args()
    return args


def get_dataset(dataset, args):
    datasets = {
        'voc': get_voc_train,
        'coco': get_coco_train
    }
    if dataset not in datasets:
        raise ValueError("dataset {} not supported".format(dataset))
    return datasets[dataset](args)


def main():
    args = parse_args()
    if not os.path.isdir(args.save_prefix):
        os.makedirs(args.save_prefix)

    config = Config(args.config_filename)
    roidb = get_dataset(args.dataset, args)
    sym = get_network(args.network, args, config, 'train')
    train_net(sym, roidb, args, config)


if __name__ == '__main__':
    main()
