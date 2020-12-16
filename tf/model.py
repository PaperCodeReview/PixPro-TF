import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.math import l2_normalize
from tensorflow.linalg import matmul
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

from resnet import ResNet50
from resnet import norm_layer


model_dict = {'resnet50': ResNet50}


def create_model(
    logger,
    backbone, 
    img_size,
    norm='bn',
    weight_decay=0.,
    lincls=False,
    classes=1000,
    snapshot=None,
    freeze=False):

    base_encoder = model_dict[backbone](
        backbone=backbone,
        input_shape=(img_size, img_size, 3),
        norm=norm,
        weight_decay=weight_decay)

    if lincls:
        # TODO
        pass
        # base_encoder.load_weights(snapshot, by_name=True)
        # if freeze:
        #     logger.info('Freeze the model!')
        #     for layer in base_encoder.layers:
        #         layer.trainable=False

        # x = Dense(classes, use_bias=False)(base_encoder.output)
        # model = Model(base_encoder.input, x, name=backbone)
        # return model

    return base_encoder


class PixPro(tf.keras.Model):
    def __init__(
        self, 
        logger, 
        backbone='resnet50',
        img_size=224,
        norm='bn',
        feature_size=7,
        channel=256,
        gamma=2., 
        num_layers=1,
        weight_decay=0.,
        snapshot=None,
        *args,
        **kwargs):

        super(PixPro, self).__init__(*args, **kwargs)
        self.norm = norm
        self.gamma = gamma

        def _get_architecture(name=None):
            base_encoder = create_model(logger, backbone, img_size, self.norm, weight_decay)
            x = base_encoder.output
            c = K.int_shape(x)[-1]

            x = Conv2D(c, 1, use_bias=False, 
                    kernel_regularizer=l2(weight_decay),
                    name='proj_conv1')(x)
            x = norm_layer(self.norm, name='proj_bn1')(x)
            x = Activation('relu', name='proj_relu1')(x)

            x = Conv2D(channel, 1, use_bias=False, 
                    kernel_regularizer=l2(weight_decay), 
                    name='proj_conv2')(x)
            x = norm_layer(self.norm, name='proj_bn2')(x)
            x = Activation('relu', name='proj_relu2')(x)

            arch = Model(base_encoder.input, x, name=name)
            return arch
        
        logger.info('Set regular encoder')
        self.encoder_regular = _get_architecture('encoder_regular')

        logger.info('Set Pixel Propagation Module')
        self.encoder_ppm = self.PPM(
            (feature_size, feature_size, channel), 
            num_layers, channel, weight_decay)
                        
        logger.info('Set propagation encoder')
        encoder_output = self.encoder_ppm(self.encoder_regular.output)
        self.encoder_propagation = Model(
            self.encoder_regular.input, encoder_output, name='propagation_encoder')

        logger.info('Set momentum encoder')
        self.encoder_momentum = _get_architecture('encoder_momentum')

        if snapshot:
            self.encoder_propagation.load_weights(snapshot)
            logger.info('Load propagation weights at {}'.format(snapshot))
            self.encoder_momentum.load_weights(snapshot.replace('/propagation/', '/momentum/'))
            logger.info('Load momentum weights at {}'.format(snapshot.replace('/propagation/', '/momentum/')))
        else:
            for i, layer in enumerate(self.encoder_regular.layers):
                self.encoder_momentum.get_layer(index=i).set_weights(layer.get_weights())

        self.encoder_momentum.trainable = False

    def PPM(self, input_shape, num_layers=1, channel=256, weight_decay=0.):
        inputs = Input(shape=input_shape)
        # transform
        if num_layers == 0:
            transform = inputs
        elif num_layers == 1:
            transform = Conv2D(channel, 1, use_bias=False,
                               kernel_regularizer=l2(weight_decay),
                               name='ppm_conv1')(inputs)
            transform = norm_layer(self.norm, name='ppm_bn1')(transform)
            transform = Activation('relu', name='ppm_relu1')(transform)
        elif num_layers == 2:
            transform = Conv2D(channel, 1, use_bias=False,
                               kernel_regularizer=l2(weight_decay),
                               name='ppm_conv1')(inputs)
            transform = norm_layer(self.norm, name='ppm_bn1')(transform)
            transform = Activation('relu', name='ppm_relu1')(transform)
            transform = Conv2D(channel, 1, use_bias=False,
                               kernel_regularizer=l2(weight_decay),
                               name='ppm_conv2')(transform)
            transform = norm_layer(self.norm, name='ppm_bn2')(transform)
            transform = Activation('relu', name='ppm_relu2')(transform)
        else:
            raise ValueError('num_layer must be lower than 3.')

        xi = Reshape((-1, channel))(inputs) # (49, 256)
        xi_norm = Lambda(lambda x: l2_normalize(x, axis=-1))(xi) # (49, 256)
        transform = Reshape((-1, channel))(transform) # (49, 256)

        # similarity computation
        sim = Lambda(lambda x: matmul(x[0], x[1], transpose_b=True))([xi_norm, xi_norm]) # (49, 49)
        sim = Activation('relu')(sim)
        sim = Lambda(lambda x: tf.pow(x, self.gamma))(sim)

        y = Lambda(lambda x: matmul(x[0], x[1]))([sim, transform]) # (49, 256)
        y = Reshape(input_shape)(y) # (7, 7, 256)
        ppm = Model(inputs, y, name='ppm')
        return ppm

    def compile(
        self,
        optimizer,
        loss,
        momentum=.99,
        num_workers=1,
        run_eagerly=None):

        super(PixPro, self).compile(optimizer=optimizer, run_eagerly=run_eagerly)
        self.loss = loss
        self.momentum = momentum
        self.num_workers = num_workers

    def pixpro_loss(self, y, x, mask, name=None):
        _, h, w, c = x.shape
        cos = self.loss(
            tf.reshape(y, (-1,h*w,c))[:,:,None,:],
            tf.reshape(tf.stop_gradient(x), (-1,h*w,c))[:,None,:,:], axis=-1)
        cos *= mask
        cos_sum = tf.reduce_sum(cos, axis=(1,2))
        mask_sum = tf.reduce_sum(mask, axis=(1,2))
        mask_cnt = tf.math.count_nonzero(mask_sum, dtype=tf.float32)
        mask_sum = tf.where(mask_sum > 0, mask_sum, 1)
        cos = cos_sum / mask_sum
        if self.num_workers > 1:
            replica_context = tf.distribute.get_replica_context()
            mask_cnt = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, mask_cnt)
        
        cos = tf.reduce_sum(cos) / mask_cnt
        return cos

    def train_step(self, data):
        view1 = data['view1']
        view2 = data['view2']
        view1_mask = data['view1_mask'] # view2 mask from view1's perspective to be applied to view2
        view2_mask = data['view2_mask'] # view1 mask from view2's perspective to be applied to view1

        x_i = self.encoder_momentum(view1, training=False)
        x_j = self.encoder_momentum(view2, training=False)
        with tf.GradientTape() as tape:
            y_i = self.encoder_propagation(view1)
            y_j = self.encoder_propagation(view2)

            cos_ij = self.pixpro_loss(y_i, x_j, view1_mask, 'cos_ij')
            cos_ji = self.pixpro_loss(y_j, x_i, view2_mask, 'cos_ji')

            loss = 2-cos_ij-cos_ji

        trainable_vars = self.encoder_propagation.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        results = {'loss': loss, 'loss_ij': cos_ij, 'loss_ji': cos_ji}
        return results