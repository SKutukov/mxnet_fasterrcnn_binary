import argparse
import ast
import pprint

import mxnet as mx
from mxnet.module import Module
import numpy as np
from tqdm import tqdm
from tools.config import Config
from symdata.bbox import im_detect
from symdata.loader import TestLoader
from symnet.logger import logger
from symnet.model import load_param, check_shape
from symdata.vis import vis_detection
import logging
from datasets.voc import get_voc_test
from datasets.coco import get_coco_test
from symnet.factory import get_network

import os

def test_net(sym, imdb, args, config):
    logger.addHandler(logging.FileHandler("{0}/{1}".format(args.prefix, 'test.log')))
    # print config
    logger.info('called with args\n{}'.format(pprint.pformat(vars(args))))

    # setup context
    ctx = mx.gpu(args.gpu)

    # load testing data
    test_data = TestLoader(imdb.roidb, batch_size=1, short=args.img_short_side, max_size=args.img_long_side,
                           mean=config.transform['img_pixel_means'], std=config.transform['img_pixel_stds'])

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

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(imdb.num_images)]
                 for _ in range(imdb.num_classes)]

    # start detection
    with tqdm(total=imdb.num_images) as pbar:
        for i, data_batch in enumerate(test_data):
            # forward
            im_info = data_batch.data[1][0]
            mod.forward(data_batch)
            rois, scores, bbox_deltas = mod.get_outputs()
            rois = rois[:, 1:]
            scores = scores[0]
            bbox_deltas = bbox_deltas[0]

            det = im_detect(rois, scores, bbox_deltas, im_info,
                            bbox_stds=args.rcnn_bbox_stds, nms_thresh=args.rcnn_nms_thresh,
                            conf_thresh=args.rcnn_conf_thresh)
            for j in range(1, imdb.num_classes):
                indexes = np.where(det[:, 0] == j)[0]
                all_boxes[j][i] = np.concatenate((det[:, -4:], det[:, [1]]), axis=-1)[indexes, :]
            pbar.update(data_batch.data[0].shape[0])


    # evaluate model
    imdb.evaluate_detections(all_boxes)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='vgg16', help='base network')
    parser.add_argument('--params', type=str, default='', help='path to trained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--imageset', type=str, default='', help='imageset splits')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device eg. 0')
    # faster rcnn params
    parser.add_argument('--img-short-side', type=int, default=600)
    parser.add_argument('--img-long-side', type=int, default=1000)
    parser.add_argument('--rcnn-batch-size', type=int, default=1)
    parser.add_argument('--rcnn-bbox-stds', type=str, default='(0.1, 0.1, 0.2, 0.2)')
    parser.add_argument('--rcnn-nms-thresh', type=float, default=0.3)
    parser.add_argument('--rcnn-conf-thresh', type=float, default=1e-3)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--config_filename', type=str)
    args = parser.parse_args()
    args.rcnn_bbox_stds = ast.literal_eval(args.rcnn_bbox_stds)
    return args

def get_dataset(dataset, args):
    datasets = {
        'voc': get_voc_test,
        'coco': get_coco_test
    }
    if dataset not in datasets:
        raise ValueError("dataset {} not supported".format(dataset))
    return datasets[dataset](args)


def main():

    args = parse_args()
    if not os.path.isdir(args.prefix):
        os.makedirs(args.prefix)

    imdb = get_dataset(args.dataset, args)
    config = Config(args.config_filename)
    sym = get_network(args.network, args, config, 'test')

    test_net(sym, imdb, args, config)


if __name__ == '__main__':
    main()