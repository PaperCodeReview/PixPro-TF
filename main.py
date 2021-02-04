import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_addons as tfa

from common import set_seed
from common import get_logger
from common import get_session
from common import get_arguments
from common import search_same
from common import create_stamp
from dataloader import set_dataset
from dataloader import DataLoader
from model import PixPro
from callback import OptionalLearningRateSchedule
from callback import create_callbacks


def train_pixpro(args, logger, initial_epoch, strategy, num_workers):
    ##########################
    # Dataset
    ##########################
    trainset = set_dataset(args.task, args.data_path)
    steps_per_epoch = args.steps or len(trainset) // args.batch_size

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== TRAINSET ==========")
    logger.info(f"    --> {len(trainset)}")
    logger.info(f"    --> {steps_per_epoch}")


    ##########################
    # Model & Generator
    ##########################
    train_generator = DataLoader(args, 'train', trainset, args.batch_size).dataloader
    with strategy.scope():
        model = PixPro(
            logger,
            norm='bn' if num_workers == 1 else 'syncbn', 
            channel=256, 
            gamma=args.gamma,
            num_layers=args.num_layers,
            snapshot=args.snapshot)
        
        if args.summary:
            model.summary()
            return

        lr_scheduler = OptionalLearningRateSchedule(
            lr=args.lr,
            lr_mode=args.lr_mode,
            lr_interval=args.lr_interval,
            lr_value=args.lr_value,
            total_epochs=args.epochs,
            steps_per_epoch=steps_per_epoch, 
            initial_epoch=initial_epoch)

        model.compile(
            # TODO : apply LARS
            optimizer=tf.keras.optimizers.SGD(lr_scheduler, momentum=.9),
            batch_size=args.batch_size,
            num_workers=num_workers,
            run_eagerly=None)


    ##########################
    # Train
    ##########################
    callbacks, initial_epoch = create_callbacks(args, logger, initial_epoch)
    if callbacks == -1:
        logger.info('Check your model.')
        return
    elif callbacks == -2:
        return

    model.fit(
        train_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,)


# TODO
def train_lincls():
    pass


def main():
    set_seed()
    args = get_arguments()
    args.lr = args.lr or 1. * args.batch_size / 256
    args, initial_epoch = search_same(args)
    if initial_epoch == -1:
        # training was already finished!
        return

    elif initial_epoch == 0:
        # first training or training with snapshot
        args.stamp = create_stamp()

    get_session(args)
    logger = get_logger("MyLogger")
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))


    ##########################
    # Strategy
    ##########################
    if len(args.gpus.split(',')) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

    num_workers = strategy.num_replicas_in_sync
    assert args.batch_size % num_workers == 0

    logger.info('{} : {}'.format(strategy.__class__.__name__, num_workers))
    logger.info("BATCH SIZE PER WORKER : {}".format(args.batch_size // num_workers))


    ##########################
    # Training
    ##########################
    if args.task == 'pretext':
        train_pixpro(args, logger, initial_epoch, strategy, num_workers)
    else:
        raise NotImplementedError()
        # train_lincls(args, logger, initial_epoch, strategy, num_workers)


if __name__ == '__main__':
    main()