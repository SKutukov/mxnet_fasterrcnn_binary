
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
        'conv5_3_bias': 'vgg0_conv12_bias'
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
        'conv5_3_bias': 'vgg0_conv9_bias'
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

def get_weight_map(step_old, is_bin_old, step_new, is_bin_new):
    return weight_map_4