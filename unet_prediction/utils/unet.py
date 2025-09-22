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


MODELS = {'UNet': UNet}