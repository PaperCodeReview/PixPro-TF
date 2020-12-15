import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def norm_layer(norm, name):
    if norm == 'bn':
        return BatchNormalization(epsilon=1.001e-5, name=name)
    elif norm == 'syncbn':
        return SyncBatchNormalization(epsilon=1.001e-5, name=name)
    else:
        raise ValueError()


def block1(x, filters, kernel_size=3, stride=1, norm='bn', weight_decay=0., 
           conv_shortcut=True, name=None):
    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, 1, strides=stride, 
                          use_bias=False,
                          kernel_regularizer=l2(weight_decay),
                          name=name+'_0_conv')(x)
        shortcut = norm_layer(norm, name+'_0_bn')(shortcut)
    else:
        shortcut = x

    x = Conv2D(filters, 1, strides=stride, 
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               name=name+'_1_conv')(x)
    x = norm_layer(norm, name=name+'_1_bn')(x)
    x = Activation('relu', name=name+'_1_relu')(x)

    x = Conv2D(filters, kernel_size, padding='SAME',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               name=name+'_2_conv')(x)
    x = norm_layer(norm, name=name+'_2_bn')(x)
    x = Activation('relu', name=name+'_2_relu')(x)

    x = Conv2D(4 * filters, 1, 
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               name=name+'_3_conv')(x)
    x = norm_layer(norm, name=name+'_3_bn')(x)

    x = Add(name=name+'_add')([shortcut, x])
    x = Activation('relu', name=name+'_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, norm='bn', weight_decay=0., name=None):
    x = block1(x, filters, stride=stride1, norm=norm, weight_decay=weight_decay, 
               name=name+'_block1')
    for i in range(2, blocks+1):
        x = block1(x, filters, norm=norm, weight_decay=weight_decay, 
                   conv_shortcut=False, name=name+'_block'+str(i))
    return x


def ResNet(stack_fn,
           preact,
           norm='bn',
           weight_decay=0.,
           backbone='resnet',
           input_shape=None,
           **kwargs):

    img_input = Input(shape=input_shape)
    
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = Conv2D(64, 7, strides=2,
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               name='conv1_conv')(x)

    if preact is False:
        x = norm_layer(norm, name='conv1_bn')(x)
        x = Activation('relu', name='conv1_relu')(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = norm_layer(norm, name='post_bn')(x)
        x = Activation('relu', name='post_relu')(x)

    model = Model(img_input, x, name=backbone)
    return model


def ResNet50(backbone,
             input_shape=None,
             norm='bn',
             weight_decay=0.,
             **kwargs):

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, norm=norm, weight_decay=weight_decay, name='conv2')
        x = stack1(x, 128, 4, norm=norm, weight_decay=weight_decay, name='conv3')
        x = stack1(x, 256, 6, norm=norm, weight_decay=weight_decay, name='conv4')
        x = stack1(x, 512, 3, norm=norm, weight_decay=weight_decay, name='conv5')
        return x
    return ResNet(stack_fn, False, norm, weight_decay, backbone, input_shape, **kwargs)