from tools.config import Config

def get_vgg16_train(args, config):
    from symnet.symbol_vgg import get_vgg_train
    if not args.pretrained:
        args.pretrained = 'model/vgg16-0000.params'
    if not args.save_prefix:
        args.save_prefix = 'model/vgg16'

    # config = Config('configs/vgg_step_{}.yml'.format(args.step))

    return get_vgg_train(anchor_scales=config.rpn['rpn_anchor_scales'], anchor_ratios=config.rpn['rpn_anchor_ratios'],
                         rpn_feature_stride=config.rpn['rpn_feat_stride'], rpn_pre_topk=config.rpn['rpn_pre_nms_topk'],
                         rpn_post_topk=config.rpn['rpn_post_nms_topk'], rpn_nms_thresh=config.rpn['rpn_nms_thresh'],
                         rpn_min_size=config.rpn['rpn_min_size'], rpn_batch_rois=args.rpn_batch_rois,
                         num_classes=config.rcnn['rcnn_num_classes'], rcnn_feature_stride=config.rcnn['rcnn_feat_stride'],
                         rcnn_pooled_size=config.rcnn['rcnn_pooled_size'], rcnn_batch_size=args.rcnn_batch_size,
                         rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=config.rcnn['rcnn_fg_fraction'],
                         rcnn_fg_overlap=config.rcnn['rcnn_fg_overlap'], rcnn_bbox_stds=config.rcnn['rcnn_bbox_stds'],
                         isBin=config.train_param['is_rcnn_top_bin'],
                         step=args.step)


def get_vgg16_test(args, config):
    from symnet.symbol_vgg import get_vgg_test
    if not args.params:
        args.params = 'model/vgg16-0010.params'

    # config = Config('configs/vgg_step_{}.yml'.format(args.step))

    return get_vgg_test(anchor_scales=config.rpn['rpn_anchor_scales'], anchor_ratios=config.rpn['rpn_anchor_ratios'],
                        rpn_feature_stride=config.rpn['rpn_feat_stride'], rpn_pre_topk=config.rpn['rpn_pre_nms_topk'],
                        rpn_post_topk=config.rpn['rpn_post_nms_topk'], rpn_nms_thresh=config.rpn['rpn_nms_thresh'],
                        rpn_min_size=config.rpn['rpn_min_size'], num_classes=config.rcnn['rcnn_num_classes'],
                        rcnn_feature_stride=config.rcnn['rcnn_feat_stride'],
                        rcnn_pooled_size=config.rcnn['rcnn_pooled_size'], rcnn_batch_size=args.rcnn_batch_size,
                        isBin=config.train_param['is_rcnn_top_bin'], step=args.step)


def get_resnet50_train(args):
    from symnet.symbol_resnet import get_resnet_train
    if not args.pretrained:
        args.pretrained = 'model/resnet-50-0000.params'
    if not args.save_prefix:
        args.save_prefix = 'model/resnet50'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    # args.net_fixed_params = ['conv0', 'stage1', 'gamma', 'beta']
    args.net_fixed_params = ['conv0', 'stage1', 'stage2', 'gamma', 'beta']

    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (14, 14)
    return get_resnet_train(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                            rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                            rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                            rpn_min_size=args.rpn_min_size, rpn_batch_rois=args.rpn_batch_rois,
                            num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                            rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                            rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=args.rcnn_fg_fraction,
                            rcnn_fg_overlap=args.rcnn_fg_overlap, rcnn_bbox_stds=args.rcnn_bbox_stds,
                            units=(3, 4, 6, 3), filter_list=(256, 512, 1024, 2048))

def get_resnet50_test(args):
    from symnet.symbol_resnet import get_resnet_test
    if not args.params:
        args.params = 'model/resnet50-0010.params'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (14, 14)
    return get_resnet_test(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                           rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                           rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                           rpn_min_size=args.rpn_min_size,
                           num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                           rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                           units=(3, 4, 6, 3), filter_list=(256, 512, 1024, 2048))

def get_resnet101_train(args, config):
    from symnet.symbol_resnet import get_resnet_train
    if not args.pretrained:
        args.pretrained = 'model/resnet-101-0000.params'
    if not args.save_prefix:
        args.save_prefix = 'model/resnet101'

    return get_resnet_train(anchor_scales=config.rpn['rpn_anchor_scales'], anchor_ratios=config.rpn['rpn_anchor_ratios'],
                            rpn_feature_stride=config.rpn['rpn_feat_stride'], rpn_pre_topk=config.rpn['rpn_pre_nms_topk'],
                            rpn_post_topk=config.rpn['rpn_post_nms_topk'], rpn_nms_thresh=config.rpn['rpn_nms_thresh'],
                            rpn_min_size=config.rpn['rpn_min_size'], rpn_batch_rois=args.rpn_batch_rois,
                            num_classes=config.rcnn['rcnn_num_classes'], rcnn_feature_stride=config.rcnn['rcnn_feat_stride'],
                            rcnn_pooled_size=config.rcnn['rcnn_pooled_size'], rcnn_batch_size=args.rcnn_batch_size,
                            rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=config.rcnn['rcnn_fg_fraction'],
                            rcnn_fg_overlap=config.rcnn['rcnn_fg_overlap'], rcnn_bbox_stds=config.rcnn['rcnn_bbox_stds'],
                            units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048),
                            step=args.step)



def get_resnet101_test(args, config):
    from symnet.symbol_resnet import get_resnet_test
    if not args.params:
        args.params = 'model/resnet101-0010.params'
    return get_resnet_test(anchor_scales=config.rpn['rpn_anchor_scales'], anchor_ratios=config.rpn['rpn_anchor_ratios'],
                           rpn_feature_stride=config.rpn['rpn_feat_stride'], rpn_pre_topk=config.rpn['rpn_pre_nms_topk'],
                           rpn_post_topk=config.rpn['rpn_post_nms_topk'], rpn_nms_thresh=config.rpn['rpn_nms_thresh'],
                           rpn_min_size=config.rpn['rpn_min_size'],
                           num_classes=config.rcnn['rcnn_num_classes'], rcnn_feature_stride=config.rcnn['rcnn_feat_stride'],
                           rcnn_pooled_size=config.rcnn['rcnn_pooled_size'], rcnn_batch_size=args.rcnn_batch_size,
                           units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048),
                           step=args.step)


def get_network(network, args, config, m_type):

    if m_type == 'train':
        networks = {
            'vgg16': get_vgg16_train,
            'resnet50': get_resnet50_train,
            'resnet101': get_resnet101_train
        }
    elif m_type == 'test':
        networks = {
            'vgg16': get_vgg16_test,
            'resnet50': get_resnet50_test,
            'resnet101': get_resnet101_test
        }
    else:
        raise ValueError('type {} not supported'.format(m_type))

    if network not in networks:
        raise ValueError("network {} not supported".format(network))
    return networks[network](args, config)
