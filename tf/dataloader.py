import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from augment import Augment


AUTO = tf.data.experimental.AUTOTUNE


def set_dataset(task, data_path):
    trainset = pd.read_csv(
        os.path.join(
            data_path, 'imagenet_trainset.csv'
        )).values.tolist()
    trainset = [[os.path.join(data_path, t[0]), t[1]] for t in trainset]

    if task == 'lincls':
        valset = pd.read_csv(
            os.path.join(
                data_path, 'imagenet_valset.csv'
            )).values.tolist()
        valset = [[os.path.join(data_path, t[0]), t[1]] for t in valset]
        return np.array(trainset, dtype='object'), np.array(valset, dtype='object')

    return np.array(trainset, dtype='object')


class DataLoader:
    def __init__(self, args, mode, datalist, batch_size, num_workers=1, shuffle=True):
        self.args = args
        self.mode = mode
        self.datalist = datalist
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.dataloader = self._dataloader()

    def __len__(self):
        return len(self.datalist)

    def fetch_dataset(self, path, y=None):
        x = tf.io.read_file(path)
        if y is not None:
            return tf.data.Dataset.from_tensors((x, y))
        return tf.data.Dataset.from_tensors(x)

    def get_distance_A(self, offset_list, size_list, isflip_list):
        '''
        offset_list : (height, width)
        size_list : (height, width)
        isflip_list : bool
        '''
        feature_size = self.args.img_size // (2**5)
        offset1, offset2 = offset_list
        size1, size2 = size_list
        isflip1, isflip2 = isflip_list

        view1_diag = tf.sqrt(tf.cast(size1[0]**2 + size1[1]**2, tf.float32))
        view2_diag = tf.sqrt(tf.cast(size2[0]**2 + size2[1]**2, tf.float32))

        def get_coordmat(offset, size, axis):
            x = tf.linspace(
                tf.cast(offset, tf.float32), 
                tf.cast(offset, tf.float32)+tf.cast(size, tf.float32), 
                feature_size)
            x = tf.expand_dims(x, axis=axis)
            x = tf.repeat(x, feature_size, axis=axis)
            return tf.cast(x, tf.float32)

        view1_x = get_coordmat(offset1[1], size1[1], 0)
        view1_y = get_coordmat(offset1[0], size1[0], 1)
        view2_x = get_coordmat(offset2[1], size2[1], 0)
        view2_y = get_coordmat(offset2[0], size2[0], 1)

        def get_distance_axis(source, target):
            d = tf.repeat(tf.reshape(source, (1, -1)), feature_size**2, axis=0)
            d -= tf.repeat(tf.reshape(target, (-1, 1)), feature_size**2, axis=1)
            return d

        view1_Ax = get_distance_axis(view1_x, view2_x)
        view1_Ay = get_distance_axis(view1_y, view2_y)
        view2_Ax = get_distance_axis(view2_x, view1_x)
        view2_Ay = get_distance_axis(view2_y, view1_y)

        view1_A = tf.sqrt(tf.square(view1_Ax)+tf.square(view1_Ay))
        view2_A = tf.sqrt(tf.square(view2_Ax)+tf.square(view2_Ay))

        view1_A_norm = view1_A / view1_diag
        view2_A_norm = view2_A / view2_diag

        view1_A_norm_mask = tf.cast(view1_A_norm < self.args.threshold, tf.float32)
        view2_A_norm_mask = tf.cast(view2_A_norm < self.args.threshold, tf.float32)
        if isflip1:
            view1_A_norm_mask = tf.reverse(view1_A_norm_mask, axis=[1])
        if isflip2:
            view2_A_norm_mask = tf.reverse(view2_A_norm_mask, axis=[1])

        return {'view1_mask': view1_A_norm_mask, 'view2_mask': view2_A_norm_mask}
        
    def augmentation(self, img, shape):
        augset = Augment(self.args, self.mode)
        if self.args.task == 'pretext':
            img_dict = {}
            offset_list = []
            size_list = []
            isflip_list = []
            prob_list = [{'p_blur': 1., 'p_solar': 0.},
                         {'p_blur': .1, 'p_solar': .2}]
            for i, view in enumerate(['view1', 'view2']): # view1, view2
                aug_img = tf.identity(img)
                aug_img, offset, size, isflip = augset._augment_pretext(aug_img, shape, **prob_list[i])
                img_dict[view] = aug_img
                offset_list.append(offset)
                size_list.append(size)
                isflip_list.append(isflip)

            A_dict = self.get_distance_A(offset_list, size_list, isflip_list)
            img_dict.update(A_dict)
            return img_dict
        else:
            raise NotImplementedError('lincls is not implemented yet.')
            # return augset(img, shape)

    def dataset_parser(self, value, label=None):
        shape = tf.image.extract_jpeg_shape(value)
        img = tf.io.decode_jpeg(value, channels=3)
        if self.args.task == 'pretext':
            return self.augmentation(img, shape)
        else:
            # lincls
            img = self.augmentation(img, shape)
            label = tf.one_hot(label, self.args.classes)
            return (img, label)
        
    def _dataloader(self):
        self.imglist = self.datalist[:,0].tolist()
        if self.args.task == 'pretext':
            dataset = tf.data.Dataset.from_tensor_slices(self.imglist)
        elif self. args.task == 'lincls':
            self.labellist = self.datalist[:,1].tolist()
            dataset = tf.data.Dataset.from_tensor_slices((self.imglist, self.labellist))
        else:
            raise NotImplementedError()

        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(len(self.datalist))

        dataset = dataset.interleave(self.fetch_dataset, num_parallel_calls=AUTO)
        dataset = dataset.map(self.dataset_parser, num_parallel_calls=AUTO)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)
        return dataset