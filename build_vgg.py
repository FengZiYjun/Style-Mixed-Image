import scipy.io
import tensorflow as tf
import numpy as np
# load VGG-19 model
VGG16_PATH = 'vgg16_conv.npz'
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
layer_names = 0

def get_weight_bias(vgg_layers, layer_i, layer_names):
    weights = vgg_layers[layer_i]
    w = tf.constant(weights)
    bias = vgg_layers[layer_i + 1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    print('get weight from ' + layer_names[layer_i])
    print('get bias from ' + layer_names[layer_i + 1])
    return w, b

def conv_relu_layer(layer_input, nwb):
    conv_val = tf.nn.conv2d(layer_input, nwb[0], strides=[1, 1, 1, 1], padding='SAME')
    relu_val = tf.nn.relu(conv_val + nwb[1])
    return relu_val

def pool_layer(pool_style, layer_input):
    if pool_style == 'avg':
        return tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    elif pool_style == 'max':
        return  tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


# net['input']   (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
# net['conv1_1']  2-D


def build(VGG16_model_path=VGG16_PATH):
    net = {}
    #vgg_rawnet = scipy.io.loadmat(VGG19_model_path, variable_names=['layers'], struct_as_record=False, squeeze_me=True)
    #vgg_layers = vgg_rawnet['layers'][0]
    npzFile = np.load(VGG16_model_path)
    layer_names = npzFile['kwds']
    vgg_layers = npzFile['args']

    net['input'] = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype('float32'))

    net['conv1_1'] = conv_relu_layer(net['input'], get_weight_bias(vgg_layers, 0, layer_names))
    net['conv1_2'] = conv_relu_layer(net['conv1_1'], get_weight_bias(vgg_layers, 2, layer_names))
    net['pool1'] = pool_layer('avg', net['conv1_2'])

    net['conv2_1'] = conv_relu_layer(net['pool1'], get_weight_bias(vgg_layers, 4, layer_names))
    net['conv2_2'] = conv_relu_layer(net['conv2_1'], get_weight_bias(vgg_layers, 6, layer_names))
    net['pool2'] = pool_layer('max', net['conv2_2'])

    net['conv3_1'] = conv_relu_layer(net['pool2'], get_weight_bias(vgg_layers, 8, layer_names))
    net['conv3_2'] = conv_relu_layer(net['conv3_1'], get_weight_bias(vgg_layers, 10, layer_names))
    net['conv3_3'] = conv_relu_layer(net['conv3_2'], get_weight_bias(vgg_layers, 12, layer_names))
    net['pool3'] = pool_layer('avg', net['conv3_3'])

    net['conv4_1'] = conv_relu_layer(net['pool3'], get_weight_bias(vgg_layers, 14, layer_names))
    net['conv4_2'] = conv_relu_layer(net['conv4_1'], get_weight_bias(vgg_layers, 16, layer_names))
    net['conv4_3'] = conv_relu_layer(net['conv4_2'], get_weight_bias(vgg_layers, 18, layer_names))
    net['pool4'] = pool_layer('max', net['conv4_3'])

    net['conv5_1'] = conv_relu_layer(net['pool4'], get_weight_bias(vgg_layers, 20, layer_names))
    net['conv5_2'] = conv_relu_layer(net['conv5_1'], get_weight_bias(vgg_layers, 22, layer_names))
    net['conv5_3'] = conv_relu_layer(net['conv5_2'], get_weight_bias(vgg_layers, 24, layer_names))
    net['pool5'] = pool_layer('avg', net['conv5_3'])

    return net


def main():
    net = build()
    print(net)
    print(net['conv1_1'].shape)

if __name__=='__main__':
    main()
