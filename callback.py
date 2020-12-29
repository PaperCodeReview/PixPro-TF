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


class PixProModelCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.save_freq == 'epoch':
            logs = logs or {}
            if isinstance(self.save_freq, 
                          int) or self.epochs_since_last_save >= self.period:
                logs = tf_utils.to_numpy_or_python_type(logs)
                self.epochs_since_last_save = 0
                filepath = self._get_file_path(epoch, logs)
                for m in self.model.layers:
                    try:
                        if 'propagation' in m.name:
                            model_name = 'propagation'
                        elif 'momentum' in m.name:
                            model_name = 'momentum'
                        else:
                            raise ValueError()

                        filepath_add = filepath.replace('checkpoint', f'checkpoint/{model_name}')
                        if self.save_best_only:
                            current = logs.get(self.monitor)
                            if current is None:
                                logging.warning('Can save best %s only with %s available, '
                                                'skipping.' % (model_name, self.monitor))
                            else:
                                if self.monitor_op(current, self.best):
                                    if self.verbose > 0:
                                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                              ' saving %s to %s' % (epoch + 1, self.monitor, self.best, 
                                                                    current, model_name, filepath_add))
                                    self.best = current
                                    if self.save_weights_only:
                                        m.save_weights(filepath_add, overwrite=True)
                                    else:
                                        m.save(filepath_add, overwrite=True)
                                else:
                                    if self.verbose > 0:
                                        print('\nEpoch %05d: %s did not improve from %0.5f' %
                                            (epoch + 1, self.monitor, self.best))
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: saving %s to %s' % (epoch + 1, model_name, filepath_add))
                            if self.save_weights_only:
                                m.save_weights(filepath_add, overwrite=True)
                            else:
                                m.save(filepath_add, overwrite=True)

                        self._maybe_remove_file()
                    except IOError as e:
                        # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                        if 'is a directory' in six.ensure_str(e.args[0]).lower():
                            raise IOError('Please specify a non-directory filepath for '
                                        'ModelCheckpoint. Filepath used is an existing '
                                        'directory: {}'.format(filepath))


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
        self.logger.info(f'Epoch {epoch:04d} Momentum : {self.momentum:.4f}')


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
        for m in ['propagation', 'momentum']:
            os.makedirs(f'{args.result_path}/{args.task}/{args.stamp}/checkpoint/{m}', exist_ok=True)
        
        if args.task in ['v1', 'v2']:
            callbacks.append(PixProModelCheckpoint(
                filepath=os.path.join(
                    f'{args.result_path}/{args.task}/{args.stamp}/checkpoint',
                    '{epoch:04d}_{loss:.4f}_{acc1:.4f}_{acc5:.4f}.h5'),
                monitor='acc1',
                mode='max',
                verbose=1,
                save_weights_only=True))
        else:
            callbacks.append(ModelCheckpoint(
                filepath=os.path.join(
                    f'{args.result_path}/{args.task}/{args.stamp}/checkpoint',
                    '{epoch:04d}_{val_loss:.4f}_{val_acc1:.4f}_{val_acc5:.4f}.h5'),
                monitor='val_acc1',
                mode='max',
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