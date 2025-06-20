#from https://github.com/williamlidberg/Detection-of-hunting-pits-using-airborne-laser-scanning-and-deep-learning
import abc
import logging
import numpy as np
import os
from enum import Enum
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers, models
import torch
from torch.optim.lr_scheduler import StepLR

from .ca_net.Models.networks.network import Comprehensive_Atten_Unet
from .ca_net.utils.dice_loss import SoftDiceLoss, get_soft_label
from .ca_net.utils.evaluation import AverageMeter


class SegmentationModelInterface(metaclass=abc.ABCMeta):

    '''Basic interface for semantic segmentation models for unified access'''
    WEIGHTING = Enum('Weighting', ['NONE', 'FOCAL', 'MANUAL'])

    @classmethod
    def focal_loss(cls, gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            target_1 = -K.mean(alpha * K.pow(1. - pt_1, gamma)
                               * K.log(pt_1+K.epsilon()))
            target_0 = - K.mean((1 - alpha) * K.pow(pt_0, gamma)
                                * K.log(1. - pt_0 + K.epsilon()))
            return target_1 + target_0
        return focal_loss_fixed

    @classmethod
    def __subclasshook__(cls, subclass):
        '''Method to check if given object implements required interface

        '''
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'load_weights') and
                callable(subclass.load_weights) and
                hasattr(subclass, 'proba') and
                callable(subclass.proba))

    @abc.abstractmethod
    def train(self, epochs, train_data, valid_data, log_path):
        '''Train segmentation model using its default parameters

        Parameters
        ----------
        epochs : Number of epochs to train for
        train_data : Iterable with the training images
        valid_data : Iterable with the validation images
        log_path : Path under which the log files and models will be stored

        '''
        raise NotImplementedError

    @abc.abstractmethod
    def load_weights(self, model_path):
        '''Load model from given weight file

        Parameters
        ----------
        model_path : Path to the file containing the model weights

        '''
        raise NotImplementedError

    @abc.abstractmethod
    def proba(self, img):
        '''Run prediction on given image(s) and return class probabilities

        Parameters
        ----------
        img : Input images of shape [batch_size, width, height, bands]

        Returns
        -------
        Output probabilities of shape [batch_size, width * height, classes]

        '''
        raise NotImplementedError


class CANet:

    '''Wrapper for the CA-Net pytorch implementation'''

    LOG_FILE = 'canet_log.csv'
    CKPT_FILE = 'minloss_checkpoint.pth.tar'
    LR_RATE = 1e-4
    WEIGTH_DECAY = 1e-8
    CHANNEL_LAST = False

    @classmethod
    def dice_score(cls, prediction, soft_ground_truth, num_class):
        ''' Compute dice score for given prediction

        Parameters
        ----------
        prediction : Predicted labels for each class
        soft_ground_truth : Ground truth values for each class
        num_class : number of classes

        Returns
        -------
        dice_mean_score : mean dice score over all classes
        dice_score : dice score per class
        '''

        pred = prediction.contiguous().view(-1, num_class)
        ground = soft_ground_truth.view(-1, num_class)

        # compute dice loss for every class
        intersect = torch.sum(ground * pred, 0)
        ref_vol = torch.sum(ground, 0)
        seg_vol = torch.sum(pred, 0)
        dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)

        dice_mean_score = torch.mean(dice_score)

        return dice_mean_score, dice_score

    def __init__(self, input_shape, depth=None, activation='softmax',
                 classes=3, entry_block=True, first_core_filters=128,
                 weighting=SegmentationModelInterface.WEIGHTING.NONE):
        '''Initialize CA-Net

        Parameters
        ----------
        input_shape : Shape of the input images
        depth : Number of downsampling and corresponding upsampling layers,
                optional
        activation : Activation function to use in the hidden layers, optional
        classes : Number of target classes, optional
        entry_block : Process input image by a CNN before starting the
                      downsampling with its separated convolutions, optional
        first_core_filters : Number of filters to use in first downsampling
                             block - determines the filter sizes in all
                             subsequent layers, optional
        weighting : Class weighting mode to use

        Returns
        -------
        Initialized model object

        '''
        self.start_epoch = 0
        self.classes = classes
        self.img_width = input_shape[1]
        self.img_height = input_shape[2]

        if weighting != SegmentationModelInterface.WEIGHTING.NONE:
            logging.warn('[CA-NET] Does not support weighting')

        in_channels = input_shape[0]
        # create dummy args object with image width and height
        args = type('obj', (object,), {'out_size': input_shape[1:]})
        canet_depth = 4 if depth is None else depth
        self.model = Comprehensive_Atten_Unet(args, in_channels, classes,
                                              canet_depth)
        self.on_gpu = False
        if torch.cuda.is_available():
            logging.info('[CA-NET] We can use %s GPUs to train the network',
                         torch.cuda.device_count())
            self.on_gpu = True
            self.model = self.model.cuda()

        # optimize all model parameters
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.LR_RATE,
                                          weight_decay=self.WEIGTH_DECAY)

    def train(self, epochs, train_data, valid_data, log_path):
        '''Train segmentation model using its default parameters

        Parameters
        ----------
        epochs : Number of epochs to train for
        train_data : Iterable with the training images
        valid_data : Iterable with the validation images
        log_path : Path under which the log files and models will be stored

        '''
        # state
        minloss = np.infty

        # Define optimizers and loss function
        criterion = SoftDiceLoss()
        scheduler = StepLR(self.optimizer, step_size=256, gamma=0.5)

        logging.info("[CA-NET] Start training ...")
        train_log = os.path.join(log_path, self.LOG_FILE)
        for epoch in range(self.start_epoch + 1, epochs + 1):
            log_vals = [epoch]
            train_avg_loss = self.__train_torch(train_data, criterion)
            log_vals.append(train_avg_loss)
            scheduler.step()

            # assess validation performance
            val_loss, val_dice, val_dices = self.__valid_torch(valid_data,
                                                               criterion)
            log_vals.append(val_loss)
            log_vals.append(val_dice)
            for dice_score in val_dices:
                log_vals.append(dice_score)

            # save best performing model
            if val_loss < minloss:
                minloss = val_loss
                model_path = os.path.join(log_path, self.CKPT_FILE)
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'opt_dict': self.optimizer.state_dict()}
                torch.save(state, model_path)

            # log epoch performance
            with open(train_log, 'a', encoding='utf-8') as log:
                log_line = ','.join(str(el) for el in log_vals)
                log.write(log_line + '\n')

    def __train_torch(self, train_data, criterion):
        losses = AverageMeter()

        self.model.train()
        for x, y in train_data:
            image = torch.from_numpy(x).float()
            # only relevant if other weigting mode is implemented
            # target = torch.from_numpy(y.reshape((-1, self.classes,
            #                                     self.img_height,
            #                                     self.img_width))).float()
            target = torch.from_numpy(y).float()
            if self.on_gpu:
                image = image.cuda()
                target = target.cuda()

            output = self.model(image)  # model output

            # get soft label -> splits single channel into one channel per
            # class -- not necessary, since the generator does this already
            # target_soft = get_soft_label(target, self.classes)
            # the dice losses
            loss = criterion(output, target, self.classes)
            losses.update(loss.data, image.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return losses.avg

    def __valid_torch(self, valid_data, criterion):
        val_losses = AverageMeter()
        val_mean_dice = AverageMeter()
        val_dices = [AverageMeter() for i in range(self.classes)]

        self.model.eval()
        for t, k in valid_data:
            image = torch.from_numpy(t).float()
            target = torch.from_numpy(k.reshape((-1, self.classes,
                                                 self.img_width,
                                                 self.img_height))).float()
            if self.on_gpu:
                image = image.cuda()
                target = target.cuda()

            # model output
            output = self.model(image)
            output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
            output_soft = get_soft_label(output_dis, self.classes)
            # get soft label
            # unnecessary
            #target_soft = get_soft_label(target, self.classes)

            # the dice losses
            val_loss = criterion(output, target, self.classes)
            val_losses.update(val_loss.data, image.size(0))

            # the dice score
            mean, scores = self.dice_score(output_soft, target,
                                           self.classes)
            val_mean_dice.update(mean.data, image.size(0))
            for score, agg in zip(scores, val_dices):
                agg.update(score.data, image.size(0))

        dice_scores = [agg.avg for agg in val_dices]
        return val_losses.avg, val_mean_dice.avg, dice_scores

    def load_weights(self, model_path):
        '''Load model from given weight file

        Parameters
        ----------
        model_path : Path to the file containing the model weights

        '''
        if self.on_gpu:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, 
                                    map_location=torch.device('cpu'))
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['opt_dict'])

    def proba(self, img):
        '''Run prediction on given image(s) and return class probabilities

        Parameters
        ----------
        img : Input images of shape [batch_size, width, height, bands]

        Returns
        -------
        Output probabilities of shape [batch_size, width * height, classes]

        '''
        if self.model.training:
            self.model.eval()

        img = torch.from_numpy(img).float()
        if self.on_gpu:
            img = img.cuda()
        prediction = self.model(img)
        # put classes last
        prediction = prediction.permute(0, 2, 3, 1)
        # reshape
        prediction = prediction.contiguous().view(img.size(0), -1,
                                                  self.classes)

        return prediction.detach().cpu()


class UNet():

    '''Basic UNet implementation'''

    CHANNEL_LAST = True

    def __init__(self, input_shape, depth=None, activation='softmax',
                 classes=2, entry_block=True, first_core_filters=128,
                 weighting=SegmentationModelInterface.WEIGHTING.FOCAL):
        '''Initialize Basic UNet

        Parameters
        ----------
        input_shape : Shape of the input images
        depth : Number of downsampling and corresponding upsampling layers,
                optional
        activation : Activation function to use in the output layer, optional
        classes : Number of target classes, optional
        entry_block : Process input image by a CNN before starting the
                      downsampling with its separated convolutions, optional
        first_core_filters : Number of filters to use in first downsampling
                             block - determines the filter sizes in all
                             subsequent layers, optional
        weighting : Class weighting mode to use

        Returns
        -------
        Initialized model object

        '''
        self.input_shape = input_shape

        if depth is not None:
            logging.warn('[UNet] does not support setting the network depth')
        if not entry_block:
            logging.warn('[UNet] does not support deactivating the entry '
                         'block')

        self.classes = classes
        self.weighting = weighting
        self.activation = activation
        self.padding = self._compute_padding(self.input_shape, 4,
                                             False)
        self.model = self.__setup_model()

    def __pad(self, size, downsampling_steps):

        div, rest = divmod(size, 2**downsampling_steps)
        if rest == 0:
            return (0, 0)

        padded = 2**downsampling_steps * (div + 1)
        padding = padded - size
        a = padding // 2
        b = padding - a
        return (a, b)

    def _compute_padding(self, input_shape, depth, entry_block):
        downsampling_steps = depth
        if entry_block:
            downsampling_steps += 1
        x, y, _ = input_shape
        l_r = self.__pad(x, downsampling_steps)
        t_b = self.__pad(y, downsampling_steps)

        return t_b, l_r

    def __setup_model(self):
        inputs = keras.Input(shape=self.input_shape)

        # add padding
        x = layers.ZeroPadding2D(padding=self.padding)(inputs)

        # Contraction path
        c1 = layers.Conv2D(32, (3, 3), activation='relu',
                           kernel_initializer='he_normal',
                           padding='same')(x)
        c1 = layers.Dropout(0.1)(c1)  # to prevent overfitting
        c1 = layers.Conv2D(32, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(p1)
        c2 = layers.Dropout(0.1)(c2)
        c2 = layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(128, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(p2)
        c3 = layers.Dropout(0.2)(c3)
        c3 = layers.Conv2D(128, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(256, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(p3)
        c4 = layers.Dropout(0.2)(c4)
        c4 = layers.Conv2D(256, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = layers.Conv2D(512, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(p4)
        c5 = layers.Dropout(0.3)(c5)
        c5 = layers.Conv2D(512, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path
        u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2),
                                    padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(256, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(u6)
        c6 = layers.Dropout(0.2)(c6)
        c6 = layers.Conv2D(256, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c6)

        u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2),
                                    padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(128, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(u7)
        c7 = layers.Dropout(0.2)(c7)
        c7 = layers.Conv2D(128, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c7)

        u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2),
                                    padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c8)

        u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2),
                                    padding='same')(c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(32, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(32, (3, 3), activation='relu',
                           kernel_initializer='he_normal', padding='same')(c9)

        outputs = layers.Conv2D(self.classes, (1, 1),
                                activation=self.activation)(c9)
        # remove padding
        outputs = layers.Cropping2D(cropping=self.padding)(outputs)
        if self.weighting == SegmentationModelInterface.WEIGHTING.MANUAL:
            # reshape to make loss weighting possible
            outputs = layers.Reshape((-1, self.classes))(outputs)

        model = keras.Model(inputs=[inputs], outputs=[outputs])
        return model

    def train(self, epochs, train_data, valid_data, log_path):
        '''Train segmentation model using its default parameters

        Parameters
        ----------
        epochs : Number of epochs to train for
        train_data : Generator with the training images
        valid_data : Generator with the validation images
        log_path : Path under which the log files and models will be stored

        '''
        metrics = ['accuracy', keras.metrics.Recall()]
        # record IoU for each class separately
        for i in range(train_data.class_num):
            metrics.append(keras.metrics.OneHotIoU(
                                            num_classes=train_data.class_num,
                                            target_class_ids=[i, ],
                                            name=f'{i}_iou'))

        if self.weighting == SegmentationModelInterface.WEIGHTING.NONE:
            self.model.compile(optimizer=keras.optimizers.Adamax(),
                               loss='categorical_crossentropy',
                               metrics=metrics)
        elif self.weighting == SegmentationModelInterface.WEIGHTING.MANUAL:
            self.model.compile(optimizer=keras.optimizers.Adamax(),
                               loss='categorical_crossentropy',
                               sample_weight_mode="temporal",
                               metrics=metrics)
        elif self.weighting == SegmentationModelInterface.WEIGHTING.FOCAL:
            self.model.compile(
                optimizer=keras.optimizers.Adamax(learning_rate=0.0001),
                loss=SegmentationModelInterface.focal_loss(gamma=2.0),
                metrics=metrics)
        else:
            raise ValueError(f'Unknown mode: {self.weighting}')

        callbacks = [
            # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10,
            #                                  mode='min'),
            keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10,
                                              min_lr=0.00001, mode='min'),
            keras.callbacks.ModelCheckpoint(
                                        os.path.join(log_path, 'trained.h5'),
                                        monitor='val_loss',
                                        save_weights_only=True,
                                        verbose=0, save_best_only=True),
            keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=5,
                                        write_grads=True, batch_size=2,
                                        write_images=True),
            keras.callbacks.CSVLogger(os.path.join(log_path, 'log.csv'),
                                      append=True, separator=';')
        ]
        self.model.fit(train_data, epochs=epochs, verbose=0,
                       callbacks=callbacks, validation_data=valid_data)

    def load_weights(self, model_path):
        '''Load model from given weight file

        Parameters
        ----------
        model_path : Path to the file containing the model weights

        '''
        self.model.load_weights(model_path)

    def proba(self, img):
        '''Run prediction on given image(s) and return class probabilities

        Parameters
        ----------
        img : Input images of shape [batch_size, width, height, bands]

        Returns
        -------
        Output probabilities of shape [batch_size, width * height, classes]

        '''
        return self.model.predict(img)


class XceptionUNet(UNet):
    # implementation adapted from:
    # https://keras.io/examples/vision/oxford_pets_image_segmentation/

    def __init__(self, input_shape, depth=None, activation='softmax',
                 classes=2, entry_block=True, first_core_filters=128,
                 weighting=SegmentationModelInterface.WEIGHTING.MANUAL):
        '''Initialize Xception Unet

        Parameters
        ----------
        input_shape : Shape of the input images
        depth : Number of downsampling and corresponding upsampling layers,
                optional
        activation : Activation function to use in the hidden layers, optional
        classes : Number of target classes, optional
        entry_block : Process input image by a CNN before starting the
                      downsampling with its separated convolutions, optional
        first_core_filters : Number of filters to use in first downsampling
                             block - determines the filter sizes in all
                             subsequent layers, optional
        weighting : Class weighting mode to use

        Returns
        -------
        Initialized model object

        '''
        self.input_shape = input_shape

        depth = 2 if depth is None else depth
        self.activation = activation
        self.classes = classes
        self.weighting = weighting
        self.entry_block = entry_block
        self.downsample_depth = 3
        self.__set_depth(depth, first_core_filters)
        self.padding = self._compute_padding(self.input_shape, depth,
                                             self.entry_block)
        self.model = self.__setup_model()

    def __set_depth(self, depth, first_core_filters):
        # setup filter list for downsampling
        start = np.log2(first_core_filters)
        start = int(start)
        self.down_sample = [2**i for i in range(start, start+depth)]
        # for deeper networks, reduce number of kernels to fit model into GPU
        # memory
        if depth >= 3:
            for i in range(2, len(self.down_sample)):
                self.down_sample[i] = self.down_sample[i] // 2

        # start downsampling with 32 filters if no CNN comes before the
        # downsampling block - keep configured depth
        if not self.entry_block:
            length = len(self.down_sample)
            self.down_sample.insert(0, 32)
            self.down_sample = self.down_sample[:length]

        # setup filter list for upsampling
        self.up_sample = self.down_sample.copy()
        self.up_sample.reverse()

        # add one more upsampling layer to compensate for initial CNN before
        # downsampling block
        if self.entry_block:
            self.up_sample.append(32)

    def __setup_model(self):
        inputs = keras.Input(shape=self.input_shape)

        # -- [First half of the network: downsampling inputs] -- #

        # add padding
        x = layers.ZeroPadding2D(padding=self.padding)(inputs)

        # Entry block
        if self.entry_block:
            x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i, filters in enumerate(self.down_sample):
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            if i < self.downsample_depth:
                x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

                # Project residual
                residual = layers.Conv2D(filters, 1, strides=2,
                                         padding="same")(
                    previous_block_activation
                )
            else:
                # do not downsample the feature maps
                residual = layers.Conv2D(filters, 1, padding="same")(
                                            previous_block_activation)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # -- [Second half of the network: upsampling inputs] -- #

        for i, filters in enumerate(self.up_sample):
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            if i >= (len(self.down_sample) - self.downsample_depth):
                x = layers.UpSampling2D(2)(x)

                # Project residual
                residual = layers.UpSampling2D(2)(previous_block_activation)
                residual = layers.Conv2D(filters, 1, padding="same")(residual)
            else:
                # no need to upsample
                residual = layers.Conv2D(filters, 1, padding="same")(
                                                previous_block_activation)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        # outputs = layers.Conv2D(self.num_classes, 3, activation="softmax",
        #                        padding="same")(x)
        outputs = layers.Conv2D(self.classes, 3, activation=self.activation,
                                padding="same")(x)
        # remove padding
        outputs = layers.Cropping2D(cropping=self.padding)(outputs)
        if self.weighting == SegmentationModelInterface.WEIGHTING.MANUAL:
            # reshape to make loss weighting possible
            outputs = layers.Reshape((-1, self.classes))(outputs)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class AttentionUNet(SegmentationModelInterface):

    '''Attention U-Net Implementation'''

    CHANNEL_LAST = True

    def __init__(self, input_shape, depth=None, activation='softmax',  # softmax for multi-class segmentation
                 classes=3, entry_block=True, first_core_filters=128,
                 weighting=SegmentationModelInterface.WEIGHTING.FOCAL):
        '''Initialize Attention U-Net
        
        Parameters
        ----------
        Same as in UNet class with minor adaptations for attention mechanism
        '''
        self.input_shape = input_shape
        self.classes = classes  # Set classes to 3 for multi-class
        self.weighting = weighting
        self.activation = activation
        self.padding = self._compute_padding(self.input_shape, 4, False)
        self.model = self.__setup_model()

    def __pad(self, size, downsampling_steps):
        div, rest = divmod(size, 2**downsampling_steps)
        if rest == 0:
            return (0, 0)
        padded = 2**downsampling_steps * (div + 1)
        padding = padded - size
        a = padding // 2
        b = padding - a
        return (a, b)

    def _compute_padding(self, input_shape, depth, entry_block):
        downsampling_steps = depth
        if entry_block:
            downsampling_steps += 1
        x, y, _ = input_shape
        l_r = self.__pad(x, downsampling_steps)
        t_b = self.__pad(y, downsampling_steps)
        return t_b, l_r

    def __setup_model(self):
        inputs = layers.Input(shape=self.input_shape)

        # Add padding
        x = layers.ZeroPadding2D(padding=self.padding)(inputs)

        # Contraction path with attention gates
        c1 = self.conv_block(x, 32)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = self.conv_block(p1, 64)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = self.conv_block(p2, 128)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = self.conv_block(p3, 256)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = self.conv_block(p4, 512)

        # Expansive path with attention gates
        u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4], axis=3)
        c6 = self.conv_block(u6, 256)

        u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3], axis=3)
        c7 = self.conv_block(u7, 128)

        u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2], axis=3)
        c8 = self.conv_block(u8, 64)

        u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = self.conv_block(u9, 32)

        # Final output layer for 3 classes (adjust this for multi-class segmentation)
        outputs = layers.Conv2D(self.classes, (1, 1), activation=self.activation)(c9)  # 3 classes
        outputs = layers.Cropping2D(cropping=self.padding)(outputs)

        model = models.Model(inputs=[inputs], outputs=[outputs])
        return model

    def conv_block(self, x, filters):
        '''Helper function for convolutional blocks'''
        c = layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        c = layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
        return layers.Dropout(0.1)(c)

def train(self, epochs, train_data, valid_data, log_path):
    '''Train segmentation model using its default parameters'''

    metrics = ['accuracy', tf.keras.metrics.Recall()]
    for i in range(train_data.class_num):
        metrics.append(tf.keras.metrics.OneHotIoU(num_classes=train_data.class_num, target_class_ids=[i, ], name=f'{i}_iou'))

    # Check weighting and compile the model
    if self.weighting == SegmentationModelInterface.WEIGHTING.NONE:
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=metrics)
    elif self.weighting == SegmentationModelInterface.WEIGHTING.MANUAL:
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', sample_weight_mode="temporal", metrics=metrics)
    elif self.weighting == SegmentationModelInterface.WEIGHTING.FOCAL:
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=SegmentationModelInterface.focal_loss(gamma=2.0), metrics=metrics)
    else:
        raise ValueError(f'Unknown mode: {self.weighting}')

    # Callbacks for training
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10, min_lr=0.00001, mode='min'),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(log_path, 'trained.h5'), monitor='val_loss', save_weights_only=True, verbose=0, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=5, write_grads=True, batch_size=2, write_images=True),
        tf.keras.callbacks.CSVLogger(os.path.join(log_path, 'log.csv'), append=True, separator=';')
    ]

    # Adjust logits shape before fitting
    def reshape_logits(y_pred):
        '''Ensure logits are reshaped correctly'''
        # Flatten logits to match label dimensions (batch_size, 250000, num_classes)
        return tf.reshape(y_pred, (-1, train_data.class_num))  # Reshape logits for compatibility

    # Create a custom loss function that reshapes the logits (if needed)
    def custom_loss(y_true, y_pred):
        y_pred = reshape_logits(y_pred)
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Optionally, replace the model loss function with the custom loss
    self.model.compile(optimizer="adam", loss=custom_loss, metrics=metrics)

    # Train the model
    self.model.fit(train_data, epochs=epochs, verbose=0, callbacks=callbacks, validation_data=valid_data)

    def load_weights(self, filepath):
        '''Load weights into the model from a specified file path'''
        self.model.load_weights(filepath)

    def proba(self, img):
        '''Run prediction on given image(s) and return class probabilities'''
        return self.model.predict(img)


MODELS = {'XceptionUNet': XceptionUNet,
          'CANet': CANet, 'UNet': UNet,
          'AttentionUNet':AttentionUNet}