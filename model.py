import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.models import Model


DEFAULT_ARGS = {
    "use_bias": False,
    "kernel_regularizer": l2(1e-5)}

BatchNorm_DICT = {
    "bn": BatchNormalization,
    "syncbn": SyncBatchNormalization}


class ResBlock(Model):
    def __init__(self, filters, kernel_size=3, stride=1, conv_shortcut=True, bn='bn', name=None, **kwargs):
        super(ResBlock, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_shortcut = conv_shortcut
        self.bn = bn
        
    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters, 1, strides=self.stride, name=self.name+'_1_conv', **DEFAULT_ARGS)
        self.bn1 = BatchNorm_DICT[self.bn](axis=-1, epsilon=1.001e-5, name=self.name+'_1_bn')
        self.conv2 = Conv2D(self.filters, self.kernel_size, padding='SAME', name=self.name+'_2_conv', **DEFAULT_ARGS)
        self.bn2 = BatchNorm_DICT[self.bn](axis=-1, epsilon=1.001e-5, name=self.name+'_2_bn')
        self.conv3 = Conv2D(4 * self.filters, 1, name=self.name+'_3_conv', **DEFAULT_ARGS)
        self.bn3 = BatchNorm_DICT[self.bn](axis=-1, epsilon=1.001e-5, name=self.name+'_3_bn')

        if self.conv_shortcut:
            self.shortcut_conv = Conv2D(4 * self.filters, 1, strides=self.stride, name=self.name+'_0_conv', **DEFAULT_ARGS)
            self.shortcut_bn = BatchNorm_DICT[self.bn](axis=-1, epsilon=1.001e-5, name=self.name+'_0_bn')

    def call(self, inputs, training=None):
        if self.conv_shortcut:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training)
        else:
            shortcut = inputs

        x = tf.nn.relu(self.bn1(self.conv1(inputs), training))
        x = tf.nn.relu(self.bn2(self.conv2(x), training))
        x = self.bn3(self.conv3(x), training)
        x = shortcut + x
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        config = super(ResBlock, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "conv_shortcut": self.conv_shortcut,
            "bn": self.bn})
        return config


class ResStack(Model):
    def __init__(self, filters, blocks, stride1=2, bn='bn', name=None, **kwargs):
        super(ResStack, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.blocks = blocks
        self.stride1 = stride1
        self.bn = bn

    def build(self, input_shape):
        self.stacks = [ResBlock(self.filters, stride=self.stride1, bn=self.bn, name=self.name+'_block1')]
        for i in range(2, self.blocks+1):
            self.stacks.append(ResBlock(self.filters, conv_shortcut=False, bn=self.bn, name=self.name+'_block'+str(i)))
        
    def call(self, inputs, training=None):
        x = inputs
        for stack in self.stacks:
            x = stack(x, training)
        return x

    def get_config(self):
        config = super(ResStack, self).get_config()
        config.update({
            "filters": self.filters, 
            "blocks": self.blocks, 
            "stride1": self.stride1,
            "bn": self.bn})
        return config
        

class ResNet50(Model):
    def __init__(self, bn='bn', **kwargs):
        super(ResNet50, self).__init__(**kwargs)
        self.bn = bn

    def build(self, input_shape):
        self.conv1 = Conv2D(64, 7, strides=2, padding='SAME', name='conv1_conv', **DEFAULT_ARGS)
        self.bn1 = BatchNorm_DICT[self.bn](axis=-1, epsilon=1.001e-5, name='conv1_bn')
        self.pool1 = MaxPooling2D(3, strides=2, padding='SAME', name='poo1_pool')

        self.resblock1 = ResStack(64, 3, stride1=1, bn=self.bn, name='conv2')
        self.resblock2 = ResStack(128, 4, bn=self.bn, name='conv3')
        self.resblock3 = ResStack(256, 6, bn=self.bn, name='conv4')
        self.resblock4 = ResStack(512, 3, bn=self.bn, name='conv5')

    def call(self, inputs, training=None):
        x = tf.nn.relu(self.bn1(self.conv1(inputs), training))
        x = self.pool1(x)
        x = self.resblock1(x, training)
        x = self.resblock2(x, training)
        x = self.resblock3(x, training)
        x = self.resblock4(x, training)
        return x

    def get_config(self):
        config = super(ResNet50, self).get_config()
        config.update({"bn": self.bn})
        return config


class Projection(Model):
    def __init__(self, channel, bn='bn', **kwargs):
        super(Projection, self).__init__(**kwargs)
        self.channel = channel
        self.bn = bn

    def build(self, input_shape):
        self.backbone = ResNet50(name='resnet50')
        self.proj_conv1 = Conv2D(input_shape[-1], 1, name='proj_conv1', **DEFAULT_ARGS)
        self.proj_bn1 = BatchNorm_DICT[self.bn](axis=-1, epsilon=1.001e-5, name='proj_bn1')
        self.proj_conv2 = Conv2D(self.channel, 1, name='proj_conv2', **DEFAULT_ARGS)

    def call(self, inputs, training=None):
        x = self.backbone(inputs, training)
        x = self.proj_conv1(x)
        x = self.proj_bn1(x, training)
        x = tf.nn.relu(x)
        x = self.proj_conv2(x)
        return x

    def get_config(self):
        config = super(Projection, self).get_config()
        config.update({"channel": self.channel, "bn": self.bn})
        return config


class PPM(Model):
    def __init__(self, num_layers, gamma=2., bn='bn', name=None, **kwargs):
        super(PPM, self).__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.gamma = gamma
        self.bn = bn

    def build(self, input_shape):
        self.transforms = []
        if self.num_layers > 0:
            self.transforms.append(('conv', Conv2D(input_shape[-1], 1, name='ppm_conv1', **DEFAULT_ARGS)))
            if self.num_layers > 1:
                self.transforms.append(('bn', BatchNorm_DICT[self.bn](name=f'ppm_bn2')))
                self.transforms.append(('conv', Conv2D(input_shape[-1], 1, name='ppm_conv2', **DEFAULT_ARGS)))

    def call(self, inputs, training=None):
        shape = K.int_shape(inputs)
        _, h, w, c = shape
        t = inputs
        for transform in self.transforms:
            if 'bn' == transform[0]:
                t = tf.nn.relu(transform[1](t, training))
            else:
                t = transform[1](t)

        t = tf.reshape(t, (-1, h*w, c))
        x = tf.reshape(inputs, (-1, h*w, c))

        sim = -cosine_similarity(x[:,:,None,:], x[:,None,:,:], axis=-1) # (-1, h*w, h*w)
        sim = tf.nn.relu(sim)
        sim = tf.pow(sim, self.gamma)

        y = tf.matmul(sim, t) # (-1, h*w, c)
        y = tf.reshape(y, (-1, h, w, c))
        return y

    def get_config(self):
        config = super(PPM, self).get_config()
        config.update({
            "num_layers" : self.num_layers,
            "gamma" : self.gamma,
            "bn": self.bn})
        return config


class PixPro(Model):
    def __init__(
        self, 
        logger, 
        norm='bn',
        img_size=224,
        channel=256,
        gamma=2., 
        num_layers=1,
        snapshot=None,
        **kwargs):

        super(PixPro, self).__init__(**kwargs)
        self.norm = norm
        self.channel = channel
        self.gamma = gamma
        self.num_layers = num_layers
        
        logger.info('Set regular encoder')
        self.encoder_regular = Projection(self.channel, bn=self.norm, name='encoder_regular')

        logger.info('Set momentum encoder')
        self.encoder_momentum = Projection(self.channel, bn=self.norm, name='encoder_momentum')

        logger.info('Set propagation encoder')
        self.ppm = PPM(self.num_layers, gamma=self.gamma, bn=self.norm, name='ppm')

        # build!
        self.encoder_regular(tf.zeros((1, img_size, img_size, 3), dtype=tf.float32), training=False)
        self.encoder_momentum(tf.zeros((1, img_size, img_size, 3), dtype=tf.float32), training=False)
        self.ppm(tf.zeros((1, img_size//(2**5), img_size//(2**5), self.channel), dtype=tf.float32), training=False)

        if snapshot:
            self.load_weights(snapshot)
            logger.info('Load weights at {}'.format(snapshot))
        else:
            for layer in self.encoder_regular.layers:
                self.encoder_momentum.get_layer(name=layer.name).set_weights(layer.get_weights())

        self.encoder_momentum.trainable = False

    def call(self, inputs, training):
        x = self.encoder_regular(inputs, training)
        x = self.ppm(x, training)
        return x

    def compile(
        self,
        optimizer,
        batch_size,
        num_workers,
        run_eagerly=None):

        super(PixPro, self).compile(
            optimizer=optimizer, run_eagerly=run_eagerly)
        self._batch_size = batch_size
        self._num_workers = num_workers

    def consistency_loss(self, y, x, mask):
        '''
        y (b, h, w, c) : feature map of the propagation encoder
        x (b, h, w, c) : featrue map of the momentum encoder
        mask (b, h*w, h*w) : mask applied to x
        '''
        _, h, w, c = x.shape
        y = tf.reshape(y, (-1, h*w, c))
        x = tf.reshape(x, (-1, h*w, c))
        cos = -cosine_similarity(y[:,:,None,:], tf.stop_gradient(x)[:,None,:,:], axis=-1) # (-1, h*w, h*w)
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

        x = self.encoder_momentum(views, training=False) # (None, 7, 7, 256)
        x_i, x_j = tf.split(x, num_or_size_splits=2, axis=0)
        with tf.GradientTape() as tape:
            y = self(views, training=True)
            y_i, y_j = tf.split(y, num_or_size_splits=2, axis=0)

            cos_ij = self.consistency_loss(y_i, x_j, view2_mask)
            cos_ji = self.consistency_loss(y_j, x_i, view1_mask)

            loss_cos = -cos_ij-cos_ji
            loss_decay = sum(self.losses)
            
            loss = loss_cos + loss_decay
            total_loss = loss / self._num_workers

        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        results = {'loss_cos': loss_cos, 'loss_ij': cos_ij, 'loss_ji': cos_ji, 'loss': loss, 'weight_decay': loss_decay}
        return results