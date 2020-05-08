from gluon_models.vgg import VGGConvBlock
import mxnet as mx
from mxnet.module import Module
from symnet.model import load_param, check_shape

def dummy_data(ctx, batch_size=1):
    return [mx.nd.random.uniform(shape=shape, ctx=ctx) for shape in ([batch_size, 3, 600, 600], [batch_size])]

data_names = ['data']
label_names = None
data_shapes = [('data', (1, 3, 1000, 600))]
label_shapes = None

data = mx.symbol.Variable(name="data")
GLUON_LAYER = VGGConvBlock(isBin=True, step=4)
GLUON_LAYER.hybridize()
conv_feat = GLUON_LAYER(data)

arg_params, aux_params = load_param("/home/skutukov/work/mxnet_fasterrcnn_binary/convert/temp-0000.params", ctx=mx.cpu())
check_shape(conv_feat, data_shapes, arg_params, aux_params)

mod = Module(conv_feat, data_names, label_names, context=mx.cpu())
mod.bind(data_shapes, label_shapes, for_training=False)
mod.init_params(arg_params=arg_params, aux_params=aux_params)


data1, _ = dummy_data(ctx=mx.cpu())
# mod.forward(data1)
mod.save_checkpoint('test_vgg', epoch=0)