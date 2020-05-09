import mxnet as mx

weight_map = {
        'conv1_1_weight': 'vgg0_conv0_weight',
        'conv1_1_bias': 'vgg0_conv0_bias',
        'conv1_2_weight': 'vgg0_conv1_weight',
        'conv1_2_bias': 'vgg0_conv1_bias',
        'conv2_1_weight': 'vgg0_conv2_weight',
        'conv2_1_bias': 'vgg0_conv2_bias',
        'conv2_2_weight': 'vgg0_conv3_weight',
        'conv2_2_bias': 'vgg0_conv3_bias',
        'conv3_1_weight': 'vgg0_conv4_weight',
        'conv3_1_bias': 'vgg0_conv4_bias',
        'conv3_2_weight': 'vgg0_conv5_weight',
        'conv3_2_bias': 'vgg0_conv5_bias',
        'conv3_3_weight': 'vgg0_conv6_weight',
        'conv3_3_bias': 'vgg0_conv6_bias',
        'conv4_1_weight': 'vgg0_conv7_weight',
        'conv4_1_bias': 'vgg0_conv7_bias',
        'conv4_2_weight': 'vgg0_conv8_weight',
        'conv4_2_bias': 'vgg0_conv8_bias',
        'conv4_3_weight': 'vgg0_conv9_weight',
        'conv4_3_bias': 'vgg0_conv9_bias',
        'conv5_1_weight': 'vgg0_conv10_weight',
        'conv5_1_bias': 'vgg0_conv10_bias',
        'conv5_2_weight': 'vgg0_conv11_weight',
        'conv5_2_bias': 'vgg0_conv11_bias',
        'conv5_3_weight': 'vgg0_conv12_weight',
        'conv5_3_bias': 'vgg0_conv12_bias',
    'fc6_weight': 'vgg1_dense0_weight',
    'fc6_bias': 'vgg1_dense0_bias',
    'fc7_weight': 'vgg1_dense1_weight',
    'fc7_bias': 'vgg1_dense1_bias',
    }

weight_map_0_1 = {
        'conv1_1_weight': 'vgg0_conv0_weight',
        'conv1_1_bias': 'vgg0_conv0_bias',
        'conv1_2_weight': 'vgg0_qconv0_weight',
        'conv1_2_bias': 'vgg0_qconv0_bias',
        'conv2_1_weight': 'vgg0_qconv1_weight',
        'conv2_1_bias': 'vgg0_qconv1_bias',
        'conv2_2_weight': 'vgg0_qconv2_weight',
        'conv2_2_bias': 'vgg0_qconv2_bias',
        'conv3_1_weight': 'vgg0_conv1_weight',
        'conv3_1_bias': 'vgg0_conv1_bias',
        'conv3_2_weight': 'vgg0_conv2_weight',
        'conv3_2_bias': 'vgg0_conv2_bias',
        'conv3_3_weight': 'vgg0_conv3_weight',
        'conv3_3_bias': 'vgg0_conv3_bias',
        'conv4_1_weight': 'vgg0_conv4_weight',
        'conv4_1_bias': 'vgg0_conv4_bias',
        'conv4_2_weight': 'vgg0_conv5_weight',
        'conv4_2_bias': 'vgg0_conv5_bias',
        'conv4_3_weight': 'vgg0_conv6_weight',
        'conv4_3_bias': 'vgg0_conv6_bias',
        'conv5_1_weight': 'vgg0_conv7_weight',
        'conv5_1_bias': 'vgg0_conv7_bias',
        'conv5_2_weight': 'vgg0_conv8_weight',
        'conv5_2_bias': 'vgg0_conv8_bias',
        'conv5_3_weight': 'vgg0_conv9_weight',
        'conv5_3_bias': 'vgg0_conv9_bias',
        'fc6_weight': 'vgg1_dense0_weight',
        'fc6_bias': 'vgg1_dense0_bias',
        'fc7_weight': 'vgg1_dense1_weight',
        'fc7_bias': 'vgg1_dense1_bias',
    }

weight_map_1_2 = {
        'vgg0_conv0_weight' :'vgg0_conv0_weight', #64 0
        'vgg0_conv0_bias'   :'vgg0_conv0_bias',   #64 0
        'vgg0_qconv0_weight':'vgg0_qconv0_weight', #64 1
        'vgg0_qconv0_bias'  :'vgg0_qconv0_bias',  #64 1
        'vgg0_qconv1_weight':'vgg0_qconv1_weight', #128 2
        'vgg0_qconv1_bias'  :'vgg0_qconv1_bias',  #128 2
        'vgg0_qconv2_weight':'vgg0_qconv2_weight', #128 3
        'vgg0_qconv2_bias'  :'vgg0_qconv2_bias',  #128 3
        'vgg0_conv1_weight' :'vgg0_qconv3_weight',  #256 4
        'vgg0_conv1_bias'   :'vgg0_qconv3_bias',   #256 4
        'vgg0_conv2_weight' :'vgg0_qconv4_weight', #256 5
        'vgg0_conv2_bias'   :'vgg0_qconv4_bias',  #256 5
        'vgg0_conv3_weight' :'vgg0_qconv5_weight', #256 6
        'vgg0_conv3_bias'   :'vgg0_qconv5_bias', #256 6
        'vgg0_conv4_weight' :'vgg0_conv1_weight', #512 7
        'vgg0_conv4_bias'   :'vgg0_conv1_bias', #512 7
        'vgg0_conv5_weight' :'vgg0_conv2_weight', #512 8
        'vgg0_conv5_bias'   :'vgg0_conv2_bias', #512 8
        'vgg0_conv6_weight' :'vgg0_conv3_weight',  #512 9
        'vgg0_conv6_bias'   :'vgg0_conv3_bias',  #512 9
        'vgg0_conv7_weight' :'vgg0_conv4_weight',  #512 10
        'vgg0_conv7_bias'   :'vgg0_conv4_bias', #512 10
        'vgg0_conv8_weight' :'vgg0_conv5_weight', #512 11
        'vgg0_conv8_bias'   :'vgg0_conv5_bias', #512 11
        'vgg0_conv9_weight' :'vgg0_conv6_weight', #512 12
        'vgg0_conv9_bias'   :'vgg0_conv6_bias', #512 12
    }


weight_map_2_3  = {
        'vgg0_conv0_weight' :'vgg0_conv0_weight', #64 0
        'vgg0_conv0_bias'   :'vgg0_conv0_bias',   #64 0
        'vgg0_qconv0_weight':'vgg0_qconv0_weight', #64 1
        'vgg0_qconv0_bias'  :'vgg0_qconv0_bias',  #64 1
        'vgg0_qconv1_weight':'vgg0_qconv1_weight', #128 2
        'vgg0_qconv1_bias'  :'vgg0_qconv1_bias',  #128 2
        'vgg0_qconv2_weight':'vgg0_qconv2_weight', #128 3
        'vgg0_qconv2_bias'  :'vgg0_qconv2_bias',  #128 3
        'vgg0_qconv3_weight' :'vgg0_qconv3_weight',  #256 4
        'vgg0_qconv3_bias'   :'vgg0_qconv3_bias',   #256 4
        'vgg0_qconv4_weight' :'vgg0_qconv4_weight', #256 5
        'vgg0_qconv4_bias'   :'vgg0_qconv4_bias',  #256 5
        'vgg0_qconv5_weight' :'vgg0_qconv5_weight', #256 6
        'vgg0_qconv5_bias'   :'vgg0_qconv5_bias', #256 6
        'vgg0_conv1_weight' :'vgg0_qconv6_weight', #512 7
        'vgg0_conv1_bias'   :'vgg0_qconv6_bias', #512 7
        'vgg0_conv2_weight' :'vgg0_qconv7_weight', #512 8
        'vgg0_conv2_bias'   :'vgg0_qconv7_bias', #512 8
        'vgg0_conv3_weight' :'vgg0_qconv8_weight',  #512 9
        'vgg0_conv3_bias'   :'vgg0_qconv8_bias',  #512 9
        'vgg0_conv4_weight' :'vgg0_conv1_weight',  #512 10
        'vgg0_conv4_bias'   :'vgg0_conv1_bias', #512 10
        'vgg0_conv5_weight' :'vgg0_conv2_weight', #512 11
        'vgg0_conv5_bias'   :'vgg0_conv2_bias', #512 11
        'vgg0_conv6_weight' :'vgg0_conv3_weight', #512 12
        'vgg0_conv6_bias'   :'vgg0_conv3_bias', #512 12
}

weight_map_3_4 = {
        'vgg0_conv0_weight' :'vgg0_conv0_weight', #64 0
        'vgg0_conv0_bias'   :'vgg0_conv0_bias',   #64 0
        'vgg0_qconv0_weight':'vgg0_qconv0_weight', #64 1
        'vgg0_qconv0_bias'  :'vgg0_qconv0_bias',  #64 1
        'vgg0_qconv1_weight':'vgg0_qconv1_weight', #128 2
        'vgg0_qconv1_bias'  :'vgg0_qconv1_bias',  #128 2
        'vgg0_qconv2_weight':'vgg0_qconv2_weight', #128 3
        'vgg0_qconv2_bias'  :'vgg0_qconv2_bias',  #128 3
        'vgg0_qconv3_weight' :'vgg0_qconv3_weight',  #256 4
        'vgg0_qconv3_bias'   :'vgg0_qconv3_bias',   #256 4
        'vgg0_qconv4_weight' :'vgg0_qconv4_weight', #256 5
        'vgg0_qconv4_bias'   :'vgg0_qconv4_bias',  #256 5
        'vgg0_qconv5_weight' :'vgg0_qconv5_weight', #256 6
        'vgg0_qconv5_bias'   :'vgg0_qconv5_bias', #256 6
        'vgg0_qconv6_weight' :'vgg0_qconv6_weight', #512 7
        'vgg0_qconv6_bias'   :'vgg0_qconv6_bias', #512 7
        'vgg0_qconv7_weight' :'vgg0_qconv7_weight', #512 8
        'vgg0_qconv7_bias'   :'vgg0_qconv7_bias', #512 8
        'vgg0_qconv8_weight' :'vgg0_qconv8_weight',  #512 9
        'vgg0_qconv8_bias'   :'vgg0_qconv8_bias',  #512 9
        'vgg0_conv1_weight' : 'vgg0_qconv9_weight',  #512 10
        'vgg0_conv1_bias'   : 'vgg0_qconv9_bias', #512 10
        'vgg0_conv2_weight' :'vgg0_qconv10_weight', #512 11
        'vgg0_conv2_bias'   :'vgg0_qconv10_bias', #512 11
        'vgg0_conv3_weight' :'vgg0_qconv11_weight', #512 12
        'vgg0_conv3_bias'   :'vgg0_qconv11_bias', #512 12
}

weight_map_4 = {
    'fc6_weight': 'vgg1_qdense0_weight',
    'fc6_bias': 'vgg1_qdense0_bias',
    'fc7_weight': 'vgg1_dense0_weight',
    'fc7_bias': 'vgg1_dense0_bias',
}

weight_map_res = {}

def load_param(params, ctx=None, weight_map=None):
    """same as mx.model.load_checkpoint, but do not load symnet and will convert context"""
    if ctx is None:
        ctx = mx.cpu()
    save_dict = mx.nd.load(params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)

        # if name in weight_map.keys():
        #     name = weight_map[name]
        if weight_map is not None:
            if name in weight_map.keys():
                name = weight_map[name]

        if tp == 'arg':
            arg_params[name] = v.as_in_context(ctx)
        if tp == 'aux':
            aux_params[name] = v.as_in_context(ctx)
    return arg_params, aux_params


def infer_param_shape(symbol, data_shapes):
    arg_shape, _, aux_shape = symbol.infer_shape(**dict(data_shapes))
    arg_shape_dict = dict(zip(symbol.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(symbol.list_auxiliary_states(), aux_shape))
    return arg_shape_dict, aux_shape_dict


def infer_data_shape(symbol, data_shapes):
    _, out_shape, _ = symbol.infer_shape(**dict(data_shapes))
    data_shape_dict = dict(data_shapes)
    out_shape_dict = dict(zip(symbol.list_outputs(), out_shape))
    return data_shape_dict, out_shape_dict


def check_shape(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    data_shape_dict, out_shape_dict = infer_data_shape(symbol, data_shapes)
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, '%s not initialized' % k
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, arg_shape_dict[k], arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, '%s not initialized' % k
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, aux_shape_dict[k], aux_params[k].shape)


def initialize_frcnn(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
    arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
    arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
    arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
    arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'])
    arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])
    arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
    arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
    arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
    arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])
    return arg_params, aux_params

def initialize_bias(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    units = {
        1: [1, 2, 3],
        2: list(range(1, 5)),
        3: list(range(1, 7)),
    }
    for i in range(1, 4):
        for j in units[i]:
            for k in [2, 3]:
                # init arg params
                # gamma_name = 'stage%d_unit%d_nb%d_gamma' % (i, j, k)
                # beta_name = 'stage%d_unit%d_nb%d_beta' % (i, j, k)
                # arg_params[gamma_name] = mx.random.normal(0.9, 1, shape=arg_shape_dict[gamma_name])
                # arg_params[beta_name] = mx.random.normal(0., 0.1, shape=arg_shape_dict[beta_name])

                # running_mean = 'stage%d_unit%d_nb%d_running_mean' % (i, j, k)
                # running_var = 'stage%d_unit%d_nb%d_running_var' % (i, j, k)
                # aux_params[running_mean] = mx.random.normal(0.0, 0.1, shape=aux_shape_dict[running_mean])
                # aux_params[running_var] = mx.random.normal(0.9, 1, shape=aux_shape_dict[running_var])
                pass
    # for i in range(0, 13):
    #     # init arg params
    #     gamma_name = 'vgg0_batchnorm%d_gamma' % i
    #     beta_name = 'vgg0_batchnorm%d_beta' % i
    #     arg_params[gamma_name] = mx.random.normal(0.8, 1, shape=arg_shape_dict[gamma_name])
    #     arg_params[beta_name] = mx.random.normal(0.4, 0.6, shape=arg_shape_dict[beta_name])
    #     # init aux params
    #     running_mean = 'vgg0_batchnorm%d_running_mean' % i
    #     running_var = 'vgg0_batchnorm%d_running_var' % i
    #     aux_params[running_mean] = mx.random.normal(0.4, 0.6, shape=aux_shape_dict[running_mean])
    #     aux_params[running_var] = mx.random.normal(0.8, 1, shape=aux_shape_dict[running_var])
    #
    # for i in  range(0, 13): #(0, 1):
    #     conv_name = 'vgg0_conv%d_weight' % i
    #     bias_name = 'vgg0_conv%d_bias' % i
    #     arg_params[conv_name] = mx.random.normal(0., 0.001, shape=arg_shape_dict[conv_name])
    #     arg_params[bias_name] = mx.random.normal(0., 0.001, shape=arg_shape_dict[bias_name])

    # for i in range(0, 11):
    #     conv_name = 'vgg0_qconv%d_weight' % i
    #     # bias_name = 'vgg0_qconv%d_bias' %i
    #     # arg_params[conv_name] = mx.random.normal(0., 0.001, shape=arg_shape_dict[conv_name])
    #     # arg_params[bias_name] = mx.random.normal(0., 0.001, shape=arg_shape_dict[bias_name])


    return arg_params, aux_params


def get_fixed_params(symbol, fixed_param_prefix=''):
    fixed_param_names = []
    if fixed_param_prefix:
        for name in symbol.list_arguments():
            for prefix in fixed_param_prefix:
                if prefix in name:
                    fixed_param_names.append(name)
    return fixed_param_names
