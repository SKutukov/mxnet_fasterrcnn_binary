from mxnet.gluon import nn
from mxnet.initializer import Xavier
import numpy as np

vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
            13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
            16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}

step_spec = {
    0: tuple(range(0, 13)),
    1: (0,) + tuple(range(4, 13)),
    2: (0,) + tuple(range(7, 13)),
    3: (0,) + tuple(range(10, 13)),
    4: (0,),
    5: (0,)
}
class VGG(nn.HybridBlock):
    def __init__(self, layers, filters, classes=1000, batch_norm=False, isBin=False, step=0, **kwargs):
        super(VGG, self).__init__(**kwargs)
        assert len(layers) == len(filters)
        with self.name_scope():
            self.features = self._make_features(layers, filters, batch_norm, step)
            self.features.add((nn.Flatten()))
            if isBin:
                self.features.add(nn.QDense(4096,
                                           weight_initializer='normal',
                                           bias_initializer='zeros'))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Dropout(rate=0.5))
                self.features.add(nn.Dense(4096,
                                           weight_initializer='normal',
                                           bias_initializer='zeros'))
                self.features.add(nn.Activation('relu'))
            else:
                self.features.add(nn.Dense(4096,
                                           weight_initializer='normal',
                                           bias_initializer='zeros'))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Dropout(rate=0.5))
                self.features.add(nn.Dense(4096,
                                           weight_initializer='normal',
                                           bias_initializer='zeros'))
                self.features.add(nn.Activation('relu'))


            self.features.add(nn.Dropout(rate=0.5))
            self.output = nn.Dense(classes,
                                   weight_initializer='normal',
                                   bias_initializer='zeros')

    def _make_features(self, layers, filters, batch_norm, step):
        featurizer = nn.HybridSequential(prefix='')

        count = 0
        for i, num in enumerate(layers):
            for _ in range(num):

                if count not in step_spec[step]:
                    conv_layer = nn.QConv2D(filters[i], kernel_size=3, padding=1,
                                             weight_initializer=Xavier(rnd_type='gaussian',
                                                                       factor_type='out',
                                                                       magnitude=2),
                                             bias_initializer='zeros',
                                             bits=1,
                                            apply_scaling=True)
                    featurizer.add(conv_layer)
                    featurizer.add(nn.Activation('relu'))
                else:
                    conv_layer = nn.Conv2D(filters[i], kernel_size=3, padding=1,
                              weight_initializer=Xavier(rnd_type='gaussian',
                                                        factor_type='out',
                                                        magnitude=2),
                              bias_initializer='zeros')
                    featurizer.add(conv_layer)
                    featurizer.add(nn.Activation('relu'))

                count += 1

                if batch_norm:
                    featurizer.add(nn.BatchNorm())

            featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_vgg(num_layers, isBin, step, **kwargs):
    layers, filters = vgg_spec[num_layers]
    net = VGG(layers, filters, isBin=isBin, step=step, **kwargs)
    return net



class VGGConvBlock(nn.HybridBlock):

    def __init__(self, isBin=False, step=0,  **kwargs):
        super(VGGConvBlock, self).__init__(**kwargs)
        base_model = get_vgg(16, isBin=isBin, step=step)
        self.features = nn.HybridSequential()
        # Exclude last 5 vgg feature layers (1 max pooling + 2 * (fc + dropout))
        for layer in base_model.features[:-8]:
            self.features.add(layer)


    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        return x

class VGGTopFeature(nn.HybridBlock):
    def __init__(self, isBin=False, step=0,  **kwargs):
        super(VGGTopFeature, self).__init__(**kwargs)
        base_model = get_vgg(16, isBin=isBin, step=step)
        self.features = nn.HybridSequential()
        # Exclude last 5 vgg feature layers (1 max pooling + 2 * (fc + dropout))
        for layer in base_model.features[-7:]:
            self.features.add(layer)


    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        return x
