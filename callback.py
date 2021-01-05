import os
import six
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.experimental import CosineDecay

from common import create_stamp


class OptionalLearningRateSchedule(LearningRateSchedule):
    def __init__(
        self, 
        lr,
        lr_mode,
        lr_interval,
        lr_value,
        total_epochs,
        steps_per_epoch, 
        initial_epoch):

        super(OptionalLearningRateSchedule, self).__init__()
        self.lr = lr
        self.lr_mode = lr_mode
        self.lr_interval = lr_interval
        self.lr_value = lr_value
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.initial_epoch = initial_epoch

        if self.lr_mode == 'exponential':
            decay_epochs = [int(e) for e in self.lr_interval.split(',')]
            lr_values = [self.lr * (self.lr_value ** k)for k in range(len(decay_epochs) + 1)]
            self.lr_scheduler = PiecewiseConstantDecay(decay_epochs, lr_values)

        elif self.lr_mode == 'cosine':
            self.lr_scheduler = CosineDecay(self.lr, self.total_epochs)

        elif self.lr_mode == 'constant':
            self.lr_scheduler = lambda x: self.lr

        else:
            raise ValueError(self.lr_mode)
            

    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'init_lr': self.lr,
            'lr_mode': self.lr_mode,
            'lr_value': self.lr_value,
            'lr_interval': self.lr_interval,}

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        step += self.initial_epoch * self.steps_per_epoch
        lr_epoch = (step / self.steps_per_epoch)
        if self.lr_mode == 'constant':
            return self.lr
        else:
            return self.lr_scheduler(lr_epoch)


class MomentumUpdate(Callback):
    def __init__(self, logger, momentum, total_epoch):
        super(MomentumUpdate, self).__init__()
        self.logger = logger
        self.init_momentum = momentum
        self.total_epoch = total_epoch

    def on_batch_end(self, batch, logs=None):
        for layer_r, layer_m in zip(self.model.encoder_regular.layers, 
                                    self.model.encoder_momentum.layers):
            r_weights = layer_r.get_weights()
            m_weights = layer_m.get_weights()
            layer_m.set_weights([m * self.momentum + r * (1.-self.momentum) for r, m in zip(r_weights, m_weights)])

    def on_epoch_begin(self, epoch, logs=None):
        self.momentum = self.init_momentum * (1 + epoch / self.total_epoch)
        self.logger.info(f'Epoch {epoch+1:04d} Momentum : {self.momentum:.4f}')


def create_callbacks(args, logger, initial_epoch):
    if not args.resume:
        if args.checkpoint or args.history or args.tensorboard:
            if os.path.isdir(f'{args.result_path}/{args.task}/{args.stamp}'):
                flag = input(f'\n{args.task}/{args.stamp} is already saved. '
                              'Do you want new stamp? (y/n) ')
                if flag == 'y':
                    args.stamp = create_stamp()
                    initial_epoch = 0
                    logger.info(f'New stamp {args.stamp} will be created.')
                elif flag == 'n':
                    return -1, initial_epoch
                else:
                    logger.info(f'You must select \'y\' or \'n\'.')
                    return -2, initial_epoch

            os.makedirs(f'{args.result_path}/{args.task}/{args.stamp}')
            yaml.dump(
                vars(args), 
                open(f'{args.result_path}/{args.task}/{args.stamp}/model_desc.yml', 'w'), 
                default_flow_style=False)
        else:
            logger.info(f'{args.stamp} is not created due to '
                        f'checkpoint - {args.checkpoint} | '
                        f'history - {args.history} | '
                        f'tensorboard - {args.tensorboard}')

    callbacks = [MomentumUpdate(logger, args.momentum, args.epochs)]
    if args.checkpoint:
        os.makedirs(f'{args.result_path}/{args.task}/{args.stamp}/checkpoint', exist_ok=True)
        callbacks.append(ModelCheckpoint(
            filepath=os.path.join(
                f'{args.result_path}/{args.task}/{args.stamp}/checkpoint',
                '{epoch:04d}_{loss:.4f}_{loss_ij:.4f}_{loss_ji:.4f}'),
            monitor='loss',
            mode='min',
            verbose=1,
            save_weights_only=True))

    if args.history:
        os.makedirs(f'{args.result_path}/{args.task}/{args.stamp}/history', exist_ok=True)
        callbacks.append(CSVLogger(
            filename=f'{args.result_path}/{args.task}/{args.stamp}/history/epoch.csv',
            separator=',', append=True))

    if args.tensorboard:
        callbacks.append(TensorBoard(
            log_dir=f'{args.result_path}/{args.task}/{args.stamp}/logs',
            histogram_freq=args.tb_histogram,
            write_graph=True, 
            write_images=True,
            update_freq=args.tb_interval,
            profile_batch=100,))

    return callbacks, initial_epoch