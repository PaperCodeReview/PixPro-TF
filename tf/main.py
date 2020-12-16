import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

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
    train_generator = DataLoader(args, 'train', trainset, args.batch_size, num_workers).dataloader
    with strategy.scope():
        model = PixPro(
            logger,
            backbone=args.backbone,
            img_size=args.img_size,
            norm='bn' if num_workers == 1 else 'syncbn',
            feature_size=7, # default of resnet50 
            channel=256,    # default of resnet50
            gamma=args.gamma,
            num_layers=args.num_layers,
            weight_decay=args.weight_decay,
            snapshot=args.snapshot)

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
            loss=tf.keras.losses.cosine_similarity,
            momentum=args.momentum,
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
# def train_lincls(args, logger, initial_epoch, strategy, num_workers):
#     assert args.snapshot is not None, 'pretrained weight is needed!'
#     ##########################
#     # Dataset
#     ##########################
#     trainset, valset = set_dataset(args.task, args.data_path)
#     steps_per_epoch = args.steps or len(trainset) // args.batch_size
#     validation_steps = len(valset) // args.batch_size

#     logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
#     logger.info("========== TRAINSET ==========")
#     logger.info(f"    --> {len(trainset)}")
#     logger.info(f"    --> {steps_per_epoch}")

#     logger.info("=========== VALSET ===========")
#     logger.info(f"    --> {len(valset)}")
#     logger.info(f"    --> {validation_steps}")


#     ##########################
#     # Model & Generator
#     ##########################
#     with strategy.scope():
#         model = create_model(
#             logger,
#             backbone=args.backbone,
#             img_size=args.img_size,
#             weight_decay=args.weight_decay,
#             lincls=True,
#             classes=args.classes,
#             snapshot=args.snapshot,
#             freeze=args.freeze)

#         lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
#         model.compile(
#             optimizer=tf.keras.optimizers.SGD(lr_scheduler, momentum=.9),
#             metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1, 'acc1', dtype=tf.float32),
#                      tf.keras.metrics.TopKCategoricalAccuracy(5, 'acc5', dtype=tf.float32)],
#             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, name='loss'))

#     train_generator = DataLoader(args, 'train', trainset, args.batch_size, num_workers).dataloader
#     val_generator = DataLoader(args, 'val', valset, args.batch_size, num_workers).dataloader


#     ##########################
#     # Train
#     ##########################
#     callbacks, initial_epoch = create_callbacks(args, logger, initial_epoch)
#     if callbacks == -1:
#         logger.info('Check your model.')
#         return
#     elif callbacks == -2:
#         return

#     model.fit(
#         train_generator,
#         validation_data=val_generator,
#         epochs=args.epochs,
#         callbacks=callbacks,
#         initial_epoch=initial_epoch,
#         steps_per_epoch=steps_per_epoch,
#         validation_steps=validation_steps)


def main():
    set_seed()
    args = get_arguments()
    args.lr = 1. * args.batch_size / 256
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