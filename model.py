import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.math import l2_normalize
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.models import Model


model_dict = {'resnet50': tf.keras.applications.ResNet50}


def create_model(
    logger,
    backbone, 
    img_size,
    norm='bn',
    weight_decay=0.,
    use_bias=False,
    lincls=False,
    classes=1000,
    snapshot=None,
    freeze=False):

    if norm == 'bn':
        custom_objects = None
        
    elif norm == 'syncbn':
        tf.keras.layers.BatchNormalization = tf.keras.layers.experimental.SyncBatchNormalization
        custom_objects = {'SyncBatchNormChangedEpsilon': tf.keras.layers.experimental.SyncBatchNormalization}
    else:
        raise ValueError()

    base_encoder = model_dict[backbone](
        include_top=False,
        pooling=None,
        weights=None,
        input_shape=(img_size, img_size, 3),
        layers=tf.keras.layers)

    if not use_bias:
        logger.info('\tConvert use_bias to False')
    if weight_decay > 0:
        logger.info(f'\tSet weight decay {weight_decay}')

    for layer in base_encoder.layers:
        if not use_bias:
            # exclude bias
            if hasattr(layer, 'use_bias'):
                setattr(layer, 'use_bias', False)
                setattr(layer, 'bias', None)
        
        if weight_decay > 0:
            # add l2 weight decay
            if hasattr(layer, 'kernel_regularizer'):
                setattr(layer, 'kernel_regularizer', l2(weight_decay))

    if weight_decay > 0:
        model_json = base_encoder.to_json()
        base_encoder = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)

    if lincls:
        # TODO
        pass
        # base_encoder.load_weights(snapshot, by_name=True)
        # if freeze:
        #     logger.info('Freeze the model!')
        #     for layer in base_encoder.layers:
        #         layer.trainable=False

        # x = Dense(classes, use_bias=True)(base_encoder.output)
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
        use_bias=False,
        snapshot=None,
        *args,
        **kwargs):

        super(PixPro, self).__init__(*args, **kwargs)
        self.norm = norm
        self.gamma = gamma

        def _get_architecture(name=None):
            base_encoder = create_model(
                logger, backbone, img_size, self.norm, weight_decay, use_bias)
            x = base_encoder.output
            c = K.int_shape(x)[-1]
            x, output = self.projection(x, c, channel, weight_decay, use_bias)
            arch = Model(base_encoder.input, [x, base_encoder.output]+output, name=name)
            return arch
        
        logger.info('Set regular encoder')
        self.encoder_regular = _get_architecture('encoder_regular')

        logger.info('Set momentum encoder')
        self.encoder_momentum = _get_architecture('encoder_momentum')

        logger.info('Set propagation encoder')
        encoder_output = self.PPM(
            self.encoder_regular.output[0],
            (feature_size, feature_size, channel), 
            num_layers, channel, weight_decay, use_bias)

        self.encoder_propagation = Model(
            self.encoder_regular.input, encoder_output+self.encoder_regular.output[1:], name='propagation_encoder')

        if snapshot:
            self.encoder_propagation.load_weights(snapshot)
            logger.info('Load propagation weights at {}'.format(snapshot))
            self.encoder_momentum.load_weights(snapshot.replace('/propagation/', '/momentum/'))
            logger.info('Load momentum weights at {}'.format(snapshot.replace('/propagation/', '/momentum/')))
        else:
            for i, layer in enumerate(self.encoder_regular.layers):
                self.encoder_momentum.get_layer(index=i).set_weights(layer.get_weights())

        self.encoder_momentum.trainable = False

    def projection(self, x, c, channel, weight_decay, use_bias):
        output = []
        x = Conv2D(c, 1, use_bias=use_bias, 
                   kernel_regularizer=l2(weight_decay),
                   name='proj_conv1')(x)
        output.append(x)
        x = tf.keras.layers.BatchNormalization(name='proj_bn1')(x)
        x = Activation('relu', name='proj_relu1')(x)
        x = Conv2D(channel, 1, use_bias=use_bias, 
                   kernel_regularizer=l2(weight_decay), 
                   name='proj_conv2')(x)
        output.append(x)
        return x, output

    def PPM(self, x, input_shape, num_layers=0, channel=256, weight_decay=0., use_bias=False):
        '''
        x : (b, h, w, c)
        '''
        transform = x
        for n in range(num_layers):
            transform = tf.keras.layers.BatchNormalization(name=f'ppm_bn{n+1}')(transform)
            transform = Activation('relu', name=f'ppm_relu{n+1}')(transform)
            transform = Conv2D(channel, 1, use_bias=use_bias,
                               kernel_regularizer=l2(weight_decay),
                               name=f'ppm_conv{n+1}')(transform)

        xi = Reshape((-1, channel))(x) # (h*w, 256)
        transform = Reshape((-1, channel))(transform) # (h*w, 256)

        # similarity computation
        sim = Lambda(lambda x: -cosine_similarity(
            tf.expand_dims(x, axis=-2), tf.expand_dims(x, axis=-3), axis=-1))(xi) # (h*w, 1, 256) x (1, h*w, 256) -> (h*w, h*w)
        sim = Activation('relu')(sim)
        sim = Lambda(lambda x: tf.pow(x, self.gamma))(sim) # (h*w, h*w)

        y = Lambda(lambda x: tf.matmul(x[0], x[1]))([sim, transform]) # (h*w, h*w) \cdot (h*w, 256) -> (h*w, 256)
        y = Reshape(input_shape)(y) # (h, w, 256)
        return [y, sim, transform]

    def compile(
        self,
        optimizer,
        batch_size,
        num_workers,
        run_eagerly=None):

        super(PixPro, self).compile(optimizer=optimizer, run_eagerly=run_eagerly)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._cnt = 0

    def consistency_loss(self, y, x, mask, name=None):
        '''
        y (b, h, w, c) : feature map of the propagation encoder
        x (b, h, w, c) : featrue map of the momentum encoder
        mask (b, h*w, h*w) : mask applied to x
        '''
        _, h, w, c = x.shape
        cos = -cosine_similarity(
            tf.expand_dims(tf.reshape(y, (-1,h*w,c)), axis=-2),                     # (-1, h*w, 1, c)
            tf.expand_dims(tf.reshape(tf.stop_gradient(x), (-1,h*w,c)), axis=-3),   # (-1, 1, h*w, c) 
            axis=-1)                                                                # -> (-1, h*w, h*w)
        cos = tf.clip_by_value(cos, -1.+K.epsilon(), 1.-K.epsilon())
        cos *= mask # (-1, h*w, h*w)
        cos = tf.reduce_sum(cos, axis=(1,2)) # (-1, 1)
        mask_cnt = tf.math.count_nonzero(mask, axis=(1,2), dtype=tf.float32) # (-1, 1)
        cos = tf.math.divide_no_nan(cos, mask_cnt) # (-1, 1)
        cos = tf.reduce_mean(cos) # (,)
        return cos

    def train_step(self, data):
        view1 = data['view1']
        view2 = data['view2']
        view1_mask = data['view1_mask']
        view2_mask = data['view2_mask']
        views = tf.concat((view1, view2), axis=0)

        x, _, _, _ = self.encoder_momentum(views, training=False) # (None, 7, 7, 256)
        x_i, x_j = tf.split(x, num_or_size_splits=2, axis=0)
        with tf.GradientTape() as tape:
            y, sim_i, transform_i, resout, proj1, proj2 = self.encoder_propagation(views)
            y_i, y_j = tf.split(y, num_or_size_splits=2, axis=0)

            cos_ij = self.consistency_loss(y_i, x_j, view2_mask, 'cos_ij')
            cos_ji = self.consistency_loss(y_j, x_i, view1_mask, 'cos_ji')

            loss_cos = -cos_ij-cos_ji
            loss_decay = sum(self.encoder_propagation.losses)
            
            loss = loss_cos / self._num_workers
            loss += loss_decay / self._num_workers

        trainable_vars = self.encoder_propagation.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        print('resout :', resout.shape, resout[0,...,1], tf.reduce_sum(resout, axis=(1,2,3)))
        print('proj1 :', proj1.shape, proj1[0,...,0])
        print('proj2 :', proj2.shape, proj2[0,...,0])
        print('x_i :', x_i.shape, x_i[0,...,0])
        print('y_i :', y_i.shape, y_i[0,...,0])
        print('transform_i :', transform_i.shape, transform_i[0,...,0])
        print('sim_i :', sim_i.shape, sim_i[0,...,0])
        print('cos_ij :', cos_ij, '\tcos_ji :', cos_ji, '\tloss :', loss_cos, '\tnorm :', loss_decay)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        results = {'loss': loss_cos, 'loss_ij': cos_ij, 'loss_ji': cos_ji, 'weight_decay': loss_decay}
        return results