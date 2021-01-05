import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


mean_std = [[0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]]

class Augment:
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.mean, self.std = mean_std

    def _augment_pretext(
        self, x, shape,
        p_blur=1.,
        p_solar=0.):

        x, offset, size = self._crop(x, shape)
        x = self._resize(x)
        x, isflip = self._random_hflip(x, p=.5)
        x = self._random_color_jitter(x, p=.8)
        x = self._random_gaussian_blur(x, p=p_blur)
        x = self._random_solarize(x, p=p_solar)
        x = self._standardize(x)
        return x, offset, size, isflip

    def _augment_lincls(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        pass
        # x = self._crop(x, shape, coord)
        # x = self._resize(x)
        # x = self._standardize(x)
        # return x

    def _standardize(self, x):
        x = tf.cast(x, tf.float32)
        x /= 255.
        x -= self.mean
        x /= self.std
        return x

    def _crop(self, x, shape, coord=[[[0., 0., 1., 1.]]]):
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            image_size=shape,
            bounding_boxes=coord,
            area_range=(.08, 1.),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)

        offset_height, offset_width, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        x = tf.slice(x, [offset_height, offset_width, 0], [target_height, target_width, 3])
        return x, (offset_height, offset_width), (target_height, target_width)

    def _resize(self, x):
        x = tf.image.resize(x, (self.args.img_size, self.args.img_size), method='bicubic')
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _color_jitter(self, x):
        _jitter_fns = [
            self._brightness,
            self._contrast,
            self._saturation,
            self._hue]
        
        for fn in _jitter_fns:
            x = fn(x)
        return x

    def _random_color_jitter(self, x, p=.8):
        if tf.less(tf.random.uniform(shape=[], dtype=tf.float32), tf.cast(p, tf.float32)):
            x = self._color_jitter(x)
        return x

    def _brightness(self, x, max_delta=.4):
        x = tf.cast(x, tf.float32)
        delta = tf.random.uniform([], 1-max_delta, 1+max_delta, dtype=tf.float32)
        x *= delta
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _contrast(self, x, contrast=.4):
        x = tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _saturation(self, x, saturation=.2):
        x = tf.image.random_saturation(x, lower=1-saturation, upper=1+saturation)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _hue(self, x, hue=.1):
        x = tf.image.random_hue(x, max_delta=hue)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _solarize(self, x, thres=128):
        thres = tf.saturate_cast(thres, x.dtype)
        return tf.where(x < thres, x, 255-x)

    def _random_hflip(self, x, p=.5):
        if tf.less(tf.random.uniform(shape=[], dtype=tf.float32), tf.cast(p, tf.float32)):
            return tf.image.flip_left_right(x), True
        return x, False

    def _random_gaussian_blur(self, x, p=.5, kernel_size=23):
        if tf.less(tf.random.uniform(shape=[], dtype=tf.float32), tf.cast(p, tf.float32)):
            x = tf.cast(x, tf.float32)
            sigma = tf.random.uniform(shape=[], minval=.1, maxval=2., dtype=tf.float32)
            radius = tf.cast(kernel_size / 2, dtype=tf.int32)
            kernel_size = radius * 2 + 1
            x_range = tf.cast(tf.range(-radius, radius+1), dtype=tf.float32)
            blur_filter = tf.exp(-tf.pow(x_range, 2.)/(2.*tf.pow(sigma, 2.)))
            blur_filter /= tf.reduce_sum(blur_filter)

            blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
            blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
            num_channels = tf.shape(x)[-1]
            blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
            blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])

            expand_batch_dim = x.shape.ndims == 3
            if expand_batch_dim:
                x = tf.expand_dims(x, axis=0)
            x = tf.nn.depthwise_conv2d(x, blur_h, strides=[1, 1, 1, 1], padding='SAME')
            x = tf.nn.depthwise_conv2d(x, blur_v, strides=[1, 1, 1, 1], padding='SAME')
            if expand_batch_dim:
                x = tf.squeeze(x, axis=0)
            return tf.saturate_cast(x, tf.uint8)
            # return tfa.image.gaussian_filter2d(x, filter_shape=filter_shape, sigma=sigma)
        return x

    def _random_solarize(self, x, p=0.):
        if tf.less(tf.random.uniform(shape=[], dtype=tf.float32), tf.cast(p, tf.float32)):
            return self._solarize(x)
        return x