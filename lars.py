"""Layer-wise Adaptive Rate Scaling optimizer for large-batch training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import tf_logging
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.training import training_ops


class LARSOptimizer(Optimizer):
    """https://people.eecs.berkeley.edu/~youyang/lars_optimizer.py"""
    """Layer-wise Adaptive Rate Scaling for large batch training.

    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)

    Implements the LARS learning rate scheme presented in the paper above. This
    optimizer is useful when scaling the batch size to up to 32K without
    significant performance degradation. It is recommended to use the optimizer
    in conjunction with:
        - Gradual learning rate warm-up
        - Linear learning rate scaling
        - Poly rule learning rate decay
    """

    def __init__(
        self,
        learning_rate,
        momentum=0.9,
        weight_decay=0.0001,
        eeta = 0.001,
        epsilon = 1e-5,
        name="LARSOptimizer",
        **kwargs):
        """Construct a new LARS Optimizer.

        Args:
        learning_rate: A `Tensor` or floating point value. The base learning rate.
        momentum: A floating point value. Momentum hyperparameter.
        weight_decay: A floating point value. Weight decay hyperparameter.
        name: Optional name prefix for variables and ops created by LARSOptimizer.

        Raises:
        ValueError: If a hyperparameter is set to a non-sensical value.
        """
        if momentum < 0.0:
            raise ValueError("momentum should be positive: %s" % momentum)

        if weight_decay < 0.0:
            raise ValueError("weight_decay should be positive: %s" % weight_decay)

        super(LARSOptimizer, self).__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        # self._learning_rate = learning_rate
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._eeta = eeta
        self._epsilon = epsilon
        self._name = name

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, "momentum")

    def _apply_dense(self, grad, var):
        # scaled_lr = self._learning_rate
        scaled_lr = self._get_hyper("learning_rate")
        decayed_grad = grad
        tf_logging.info("LARS: apply dense: %s", var.name)
        if 'batch_normalization' not in var.name and 'bias' not in var.name:
            tf_logging.info("LARS: apply dense, decay: %s", var.name)
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(grad, ord=2)
            trust_ratio = tf.where(
                tf.math.greater(w_norm, 0),
                tf.where(
                    tf.math.greater(g_norm, 0),
                    (self._eeta * w_norm /
                    (g_norm + self._weight_decay * w_norm + self._epsilon)),
                    1.0),
                1.0)
            trust_ratio = tf.clip_by_value(trust_ratio, 0.0, 50)
            scaled_lr *= trust_ratio
            decayed_grad = grad  + self._weight_decay * var

        decayed_grad = tf.clip_by_value(decayed_grad, -10.0, 10.0)
        mom = self.get_slot(var, "momentum")
        return training_ops.apply_momentum(
            var, mom,
            scaled_lr,
            decayed_grad,
            self._momentum,
            use_locking=False,
            use_nesterov=False)

    def _resource_apply_dense(self, grad, var):
        # scaled_lr = self._learning_rate
        scaled_lr = self._get_hyper("learning_rate")
        print(scaled_lr)
        decayed_grad = grad
        tf_logging.info("LARS: resouce apply dense: %s", var.name)
        w_norm = tf.norm(var, ord=2)
        g_norm = tf.norm(grad, ord=2)
        if 'batch_normalization' not in var.name and 'bias' not in var.name:
            tf_logging.info("LARS: apply dense, decay: %s", var.name)
            trust_ratio = tf.where(
                tf.math.greater(w_norm, 0),
                tf.where(
                    tf.math.greater(g_norm, 0),
                    (self._eeta * w_norm /
                    (g_norm + self._weight_decay * w_norm + self._epsilon)),
                    1.0),
                1.0)
            trust_ratio = tf.clip_by_value(trust_ratio, 0.0, 50)
            scaled_lr *= trust_ratio
            decayed_grad = grad + self._weight_decay * var

        decayed_grad = tf.clip_by_value(decayed_grad, -10.0, 10.0)
        mom = self.get_slot(var, "momentum")
        return training_ops.resource_apply_momentum(
            var.handle, mom.handle,
            scaled_lr,
            decayed_grad,
            self._momentum,
            use_locking=False,
            use_nesterov=False)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self._serialize_hyperparameter("momentum"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "eeta": self._serialize_hyperparameter("eeta"),
                "epsilon": self.epsilon,
            }
        )
        return config