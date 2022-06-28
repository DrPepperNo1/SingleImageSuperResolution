import tensorflow as tf
import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)
from tensorflow.keras import datasets, layers, models
from keras.layers.core import Activation
from keras import metrics
from keras.layers.convolutional import Conv2D
import matplotlib as plt
import keras
from keras.models import load_model
import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from keras.models import Model
from keras.layers.core import Dense
from PIL import Image

tf.config.run_functions_eagerly(True)  # .numpy()启用


class SR_LUT_CNN():
    def __init__(self):
        self.min_x_train = 0
        self.max_x_train = 2160
        self.min_y_train = 0
        self.max_y_train = 2880
        self.min_x_val = 0
        self.max_x_val = 2160
        self.min_y_val = 2880
        self.max_y_val = 3840
        model = models.Sequential()
        model.add(Conv2D(128, (2, 2), batch_input_shape=(None, 2, 2, 1)))
        # model.add(layers.BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, (1, 1)))
        # model.add(layers.BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, (1, 1)))
        ##model.add(layers.BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (1, 1)))
        ##model.add(layers.BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (1, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (1, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (1, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (1, 1)))
        model.compile(optimizer='adam', loss=self.my_loss, metrics=metrics.mse)
        # model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
        model.summary()
        self.model = model

    def my_loss(self, y_true, y_pred):

        # print('pred is {}'.format(y_pred.numpy().shape))
        # print('true is {}'.format(y_true.numpy().shape))
        '''
        y_pred_np = y_pred.numpy()
        y_true_np = y_true.numpy()
        sum_pred = np.zeros((4,16))
        sum = 0
        for i in range(4):
            for j in range(16):
                sum = sum + (y_pred_np[i][0][0][j] - y_true_np[j])**2
        print(sum/64)

        print((tf.math.reduce_sum(tf.math.square(y_pred - y_true)) / 64).numpy())
        '''
        return tf.math.reduce_sum(tf.math.square(y_pred - y_true)) / 11520

    def Train(self, path_LR, path_HR, whichchannel=0):
        if whichchannel == 0:
            filepath = 'b_channel.hdf5'
        if whichchannel == 1:
            filepath = 'g_channel.hdf5'
        if whichchannel == 2:
            filepath = 'r_channel.hdf5'

        train_gen = self.Image_Data_Onechannel(min_x=self.min_x_train, max_x=self.max_x_train, min_y=self.min_y_train,
                                               max_y=self.max_y_train,
                                               path_LR=path_LR,
                                               path_HR=path_HR,
                                               whichchannel=whichchannel)
        val_gen = self.Image_Data_Onechannel(min_x=self.min_x_val, max_x=self.max_x_val, min_y=self.min_y_val,
                                             max_y=self.max_y_val,
                                             path_LR=path_LR,
                                             path_HR=path_HR,
                                             whichchannel=whichchannel)
        train_steps = (self.max_x_train - self.min_x_train) * (self.max_y_train - self.min_y_train) * 0.25 * 0.25
        val_steps = (self.max_x_val - self.min_x_val) * (self.max_y_val - self.min_y_val) * 0.25 * 0.25
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                       verbose=1,
                                                       monitor='val_loss',
                                                       save_weights_only=False,
                                                       mode='min',
                                                       save_best_only=True,
                                                       factor=0.1,
                                                       patience=3,
                                                       epsilon=1e-4)

        history = self.model.fit_generator(train_gen,
                                           steps_per_epoch=train_steps,
                                           epochs=1,
                                           validation_data=val_gen,
                                           validation_steps=val_steps,
                                           callbacks=[checkpointer])

    def Train_batch(self, path_LR, path_HR, whichchannel=0):
        if whichchannel == 0:
            filepath = 'b_channel.hdf5'
        if whichchannel == 1:
            filepath = 'g_channel.hdf5'
        if whichchannel == 2:
            filepath = 'r_channel.hdf5'

        train_gen = self.Image_Data_Onechannel_batch(min_x=self.min_x_train, max_x=self.max_x_train,
                                                     min_y=self.min_y_train,
                                                     max_y=self.max_y_train,
                                                     path_LR=path_LR,
                                                     path_HR=path_HR,
                                                     whichchannel=whichchannel)
        val_gen = self.Image_Data_Onechannel_batch(min_x=self.min_x_val, max_x=self.max_x_val, min_y=self.min_y_val,
                                                   max_y=self.max_y_val,
                                                   path_LR=path_LR,
                                                   path_HR=path_HR,
                                                   whichchannel=whichchannel)
        train_steps = (self.max_x_train - self.min_x_train) * (self.max_y_train - self.min_y_train) * 0.25 * 0.25 / 180
        val_steps = (self.max_x_val - self.min_x_val) * (self.max_y_val - self.min_y_val) * 0.25 * 0.25 / 180
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                       verbose=1,
                                                       monitor='val_loss',
                                                       save_weights_only=False,
                                                       mode='min',
                                                       save_best_only=True,
                                                       factor=0.1,
                                                       patience=4,
                                                       epsilon=1e-4)

        history = self.model.fit_generator(train_gen,
                                           steps_per_epoch=train_steps,
                                           epochs=10,
                                           validation_data=val_gen,
                                           validation_steps=val_steps,
                                           callbacks=[checkpointer])

    def Train_batch_rotation(self, path_LR, path_HR, whichchannel=0):
        if whichchannel == 0:
            filepath = 'b_channel.hdf5'
        if whichchannel == 1:
            filepath = 'g_channel.hdf5'
        if whichchannel == 2:
            filepath = 'r_channel.hdf5'

        train_gen = self.Image_Data_Onechannel_batch_rotation(min_x=self.min_x_train + 4, max_x=self.max_x_train - 4,
                                                              min_y=self.min_y_train + 4,
                                                              max_y=self.max_y_train - 4,
                                                              path_LR=path_LR,
                                                              path_HR=path_HR,
                                                              whichchannel=whichchannel)
        val_gen = self.Image_Data_Onechannel_batch_rotation(min_x=self.min_x_val + 4, max_x=self.max_x_val - 4,
                                                            min_y=self.min_y_val + 4,
                                                            max_y=self.max_y_val - 4,
                                                            path_LR=path_LR,
                                                            path_HR=path_HR,
                                                            whichchannel=whichchannel)
        train_steps = (self.max_x_train - self.min_x_train - 8) * (
                    self.max_y_train - self.min_y_train - 8) * 0.25 * 0.25 / 180
        val_steps = (self.max_x_val - self.min_x_val - 8) * (self.max_y_val - self.min_y_val - 8) * 0.25 * 0.25 / 180
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                       verbose=1,
                                                       monitor='val_loss',
                                                       save_weights_only=False,
                                                       mode='min',
                                                       save_best_only=True,
                                                       factor=0.1,
                                                       patience=4,
                                                       epsilon=1e-4)

        history = self.model.fit_generator(train_gen,
                                           steps_per_epoch=train_steps,
                                           epochs=2,
                                           validation_data=val_gen,
                                           validation_steps=val_steps,
                                           callbacks=[checkpointer])

    def Image_Data_Onechannel_batch(self, min_x, max_x, min_y, max_y, path_LR, path_HR, whichchannel=0):
        batch_size = 180
        Img_LR = cv2.imread(path_LR, flags=1)
        Img_LR = cv2.copyMakeBorder(Img_LR, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        # Img_LR = Img_LR / 255
        Img_HR = cv2.imread(path_HR, flags=1)
        # Img_HR = Img_HR / 255
        b_LR, g_LR, r_LR = cv2.split(Img_LR)
        b_HR, g_HR, r_HR = cv2.split(Img_HR)
        patch = np.zeros((2, 2, 1))
        true_flatten = np.zeros((16, 1))
        true_flatten_batch = np.zeros((batch_size, 1, 1, 16))
        input = np.zeros((batch_size, 2, 2, 1))
        # inputhh = np.zeros((batch_size, 30, 30, 1))
        while 1:
            num = 0
            for i in range(min_x, max_x, 4):
                for j in range(min_y, max_y, 4):
                    '''
                        00*01 *** 2*3
                        ***** *** ***
                        10*11 *** 1*0
                    '''
                    if whichchannel == 0:
                        split_LR = b_LR
                        split_HR = b_HR
                    if whichchannel == 1:
                        split_LR = g_LR
                        split_HR = g_HR
                    if whichchannel == 2:
                        split_LR = r_LR
                        split_HR = r_HR
                    if num < batch_size:
                        num = num + 1
                        # rotation 0degree
                        patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[0][1] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                        patch[1][0] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                        patch[1][1] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 1]
                        input[num - 1] = patch
                        for m in range(4):
                            for n in range(4):
                                true_flatten[m * 4 + n] = split_HR[i + m][j + n]
                        true_flatten_batch[num - 1] = true_flatten.reshape((1, 1, 16))
                    else:
                        yield input, true_flatten_batch
                        num = 1
                        patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[0][1] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                        patch[1][0] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                        patch[1][1] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 1]
                        input[num - 1] = patch
                        for m in range(4):
                            for n in range(4):
                                true_flatten[m * 4 + n] = split_HR[i + m][j + n]
                        true_flatten_batch[num - 1] = true_flatten.reshape((1, 1, 16))

    def Image_Data_Onechannel_batch_rotation(self, min_x, max_x, min_y, max_y, path_LR, path_HR, whichchannel=0):
        batch_size = 180 * 4
        Img_LR = cv2.imread(path_LR, flags=1)
        Img_LR = cv2.copyMakeBorder(Img_LR, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        # Img_LR = Img_LR / 255
        Img_HR = cv2.imread(path_HR, flags=1)
        # Img_HR = Img_HR / 255
        b_LR, g_LR, r_LR = cv2.split(Img_LR)
        b_HR, g_HR, r_HR = cv2.split(Img_HR)
        patch = np.zeros((2, 2, 1))
        true_flatten = np.zeros((16, 1))
        true_flatten_batch = np.zeros((batch_size, 1, 1, 16))
        input = np.zeros((batch_size, 2, 2, 1))

        # inputhh = np.zeros((batch_size, 30, 30, 1))
        while 1:
            num = 0
            for i in range(min_x, max_x, 4):
                for j in range(min_y, max_y, 4):
                    '''
                        00*01 *** 2*3
                        ***** *** ***
                        10*11 *** 1*0
                    '''
                    if whichchannel == 0:
                        split_LR = b_LR
                        split_HR = b_HR
                    if whichchannel == 1:
                        split_LR = g_LR
                        split_HR = g_HR
                    if whichchannel == 2:
                        split_LR = r_LR
                        split_HR = r_HR
                    if num < batch_size / 4:
                        num = num + 1
                        # rotation 0degree
                        patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[0][1] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                        patch[1][0] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                        patch[1][1] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 1]
                        input[4 * (num - 1) + 0] = patch

                        # rotation 90 degree
                        patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[0][1] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) + 0]
                        patch[1][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                        patch[1][1] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) + 1]
                        input[4 * (num - 1) + 1] = patch
                        # rotation 180 degree
                        patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[0][1] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) - 1]
                        patch[1][0] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) + 0]
                        patch[1][1] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) - 1]
                        input[4 * (num - 1) + 2] = patch
                        # rotation 270 degree
                        patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[0][1] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                        patch[1][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) - 1]
                        patch[1][1] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) - 1]
                        input[4 * (num - 1) + 3] = patch

                        true = split_HR[i - 3:i + 4, j - 3:j + 4]
                        true90 = np.rot90(true, -1)
                        true180 = np.rot90(true, -2)
                        true270 = np.rot90(true, -3)
                        true_flatten_batch[4 * (num - 1) + 0] = true[3:7, 3:7].reshape((1, 1, 16))
                        true_flatten_batch[4 * (num - 1) + 1] = true90[3:7, 3:7].reshape((1, 1, 16))
                        true_flatten_batch[4 * (num - 1) + 2] = true180[3:7, 3:7].reshape((1, 1, 16))
                        true_flatten_batch[4 * (num - 1) + 3] = true270[3:7, 3:7].reshape((1, 1, 16))

                    else:
                        yield input, true_flatten_batch
                        num = 1
                        # rotation 0degree
                        patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[0][1] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                        patch[1][0] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                        patch[1][1] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 1]
                        input[4 * (num - 1) + 0] = patch

                        # rotation 90 degree
                        patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[0][1] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) + 0]
                        patch[1][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                        patch[1][1] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) + 1]
                        input[4 * (num - 1) + 1] = patch
                        # rotation 180 degree
                        patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[0][1] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) - 1]
                        patch[1][0] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) + 0]
                        patch[1][1] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) - 1]
                        input[4 * (num - 1) + 2] = patch
                        # rotation 270 degree
                        patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[0][1] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                        patch[1][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) - 1]
                        patch[1][1] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) - 1]
                        input[4 * (num - 1) + 3] = patch
                        true = split_HR[i - 3:i + 4, j - 3:j + 4]
                        true90 = np.rot90(true, -1)
                        true180 = np.rot90(true, -2)
                        true270 = np.rot90(true, -3)
                        true_flatten_batch[4 * (num - 1) + 0] = true[3:7, 3:7].reshape((1, 1, 16))
                        true_flatten_batch[4 * (num - 1) + 1] = true90[3:7, 3:7].reshape((1, 1, 16))
                        true_flatten_batch[4 * (num - 1) + 2] = true180[3:7, 3:7].reshape((1, 1, 16))
                        true_flatten_batch[4 * (num - 1) + 3] = true270[3:7, 3:7].reshape((1, 1, 16))

    def Image_Data_Onechannel(self, min_x, max_x, min_y, max_y, path_LR, path_HR, whichchannel=0):
        Img_LR = cv2.imread(path_LR, flags=1)
        Img_LR = cv2.copyMakeBorder(Img_LR, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        # Img_LR = Img_LR / 255
        Img_HR = cv2.imread(path_HR, flags=1)
        # Img_HR = Img_HR / 255
        b_LR, g_LR, r_LR = cv2.split(Img_LR)
        b_HR, g_HR, r_HR = cv2.split(Img_HR)
        patch = np.zeros((2, 2, 1))
        true = np.zeros((4, 4, 1))
        true_flatten = np.zeros((16, 1))
        true_flatten_batch = np.zeros((4, 1, 1, 16))
        input = np.zeros((4, 2, 2, 1))
        while 1:
            for i in range(min_x, max_x, 4):
                for j in range(min_y, max_y, 4):
                    '''
                        00*01 *** 2*3
                        ***** *** ***
                        10*11 *** 1*0
                    '''
                    if whichchannel == 0:
                        split_LR = b_LR
                        split_HR = b_HR
                    if whichchannel == 1:
                        split_LR = g_LR
                        split_HR = g_HR
                    if whichchannel == 2:
                        split_LR = r_LR
                        split_HR = r_HR
                    # rotation 0degree
                    patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                    patch[0][1] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                    patch[1][0] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                    patch[1][1] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 1]
                    input[0] = patch

                    # rotation 90 degree
                    patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                    patch[0][1] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) + 0]
                    patch[1][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                    patch[1][1] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) + 1]
                    input[1] = patch
                    # rotation 180 degree
                    patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                    patch[0][1] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) - 1]
                    patch[1][0] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) + 0]
                    patch[1][1] = split_LR[1 + int(i / 4) - 1][1 + int(j / 4) - 1]
                    input[2] = patch
                    # rotation 270 degree
                    patch[0][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                    patch[0][1] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                    patch[1][0] = split_LR[1 + int(i / 4) + 0][1 + int(j / 4) - 1]
                    patch[1][1] = split_LR[1 + int(i / 4) + 1][1 + int(j / 4) - 1]
                    input[3] = patch

                    for m in range(4):
                        for n in range(4):
                            true[m][n] = split_HR[i + m][j + n]
                            true_flatten[m * 4 + n] = split_HR[i + m][j + n]
                    true_flatten_batch[0] = true_flatten.reshape((1, 1, 16))
                    true_flatten_batch[1] = true_flatten.reshape((1, 1, 16))
                    true_flatten_batch[2] = true_flatten.reshape((1, 1, 16))
                    true_flatten_batch[3] = true_flatten.reshape((1, 1, 16))
                    yield input, true_flatten_batch

    def Train_continue(self, path_model_b, path_model_g, path_model_r, rootpath_LR, rootpath_HR):
        '''

        :param path_model_b: b通道模型路径
        :param path_model_g: g通道模型路径
        :param path_model_r: r通道模型路径
        :param rootpath_LR: 低分辨率图像根目录，必须以\\结尾
        :param rootpath_HR: 高分辨率图像根目录，必须以\\结尾
        :return:
        '''
        # 读取文件名
        name_LR = os.listdir(rootpath_LR)
        name_HR = os.listdir(rootpath_HR)
        train_steps = (self.max_x_train - self.min_x_train) * (self.max_y_train - self.min_y_train) * 0.25 * 0.25
        val_steps = (self.max_x_val - self.min_x_val) * (self.max_y_val - self.min_y_val) * 0.25 * 0.25
        for name in name_LR[1:]:
            # 导入已训练模型
            b_model = load_model(path_model_b, compile=False)
            b_model.compile(optimizer='adam', loss=self.my_loss, metrics=metrics.mse)
            g_model = load_model(path_model_g, compile=False)
            g_model.compile(optimizer='adam', loss=self.my_loss, metrics=metrics.mse)
            r_model = load_model(path_model_r, compile=False)
            r_model.compile(optimizer='adam', loss=self.my_loss, metrics=metrics.mse)
            for whichchannel in range(3):
                if whichchannel == 0:
                    model = b_model
                    filepath = 'b_channel.hdf5'
                if whichchannel == 1:
                    model = g_model
                    filepath = 'g_channel.hdf5'
                if whichchannel == 2:
                    model = r_model
                    filepath = 'r_channel.hdf5'

                train_gen = self.Image_Data_Onechannel(min_x=self.min_x_train, max_x=self.max_x_train,
                                                       min_y=self.min_y_train,
                                                       max_y=self.max_y_train,
                                                       path_LR=rootpath_LR + name,
                                                       path_HR=rootpath_HR + name,
                                                       whichchannel=whichchannel)

                val_gen = self.Image_Data_Onechannel(min_x=self.min_x_val, max_x=self.max_x_val,
                                                     min_y=self.min_y_val,
                                                     max_y=self.max_y_val,
                                                     path_LR=rootpath_LR + name,
                                                     path_HR=rootpath_HR + name,
                                                     whichchannel=whichchannel)

                checkpointer = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                               verbose=1,
                                                               monitor='val_loss',
                                                               save_weights_only=False,
                                                               mode='min',
                                                               save_best_only=True,
                                                               factor=0.1,
                                                               patience=2,
                                                               epsilon=1e-4)

                history = model.fit_generator(train_gen,
                                              steps_per_epoch=train_steps,
                                              epochs=1,
                                              validation_data=val_gen,
                                              validation_steps=val_steps,
                                              callbacks=[checkpointer])
                del history
            print(name)

    def Train_continue_batch(self, path_model_b, path_model_g, path_model_r, rootpath_LR, rootpath_HR):
        '''

        :param path_model_b: b通道模型路径
        :param path_model_g: g通道模型路径
        :param path_model_r: r通道模型路径
        :param rootpath_LR: 低分辨率图像根目录，必须以\\结尾
        :param rootpath_HR: 高分辨率图像根目录，必须以\\结尾
        :return:
        '''
        # 读取文件名
        name_LR = os.listdir(rootpath_LR)
        name_HR = os.listdir(rootpath_HR)
        train_steps = (self.max_x_train - self.min_x_train) * (self.max_y_train - self.min_y_train) * 0.25 * 0.25 / 180
        val_steps = (self.max_x_val - self.min_x_val) * (self.max_y_val - self.min_y_val) * 0.25 * 0.25 / 180
        for name in name_LR[1:]:
            # 导入已训练模型
            b_model = load_model(path_model_b, compile=False)
            b_model.compile(optimizer='adam', loss=self.my_loss, metrics=metrics.mse)
            g_model = load_model(path_model_g, compile=False)
            g_model.compile(optimizer='adam', loss=self.my_loss, metrics=metrics.mse)
            r_model = load_model(path_model_r, compile=False)
            r_model.compile(optimizer='adam', loss=self.my_loss, metrics=metrics.mse)
            for whichchannel in range(3):
                if whichchannel == 0:
                    model = b_model
                    filepath = 'b_channel.hdf5'
                if whichchannel == 1:
                    model = g_model
                    filepath = 'g_channel.hdf5'
                if whichchannel == 2:
                    model = r_model
                    filepath = 'r_channel.hdf5'

                train_gen = self.Image_Data_Onechannel_batch(min_x=self.min_x_train, max_x=self.max_x_train,
                                                             min_y=self.min_y_train,
                                                             max_y=self.max_y_train,
                                                             path_LR=rootpath_LR + name,
                                                             path_HR=rootpath_HR + name,
                                                             whichchannel=whichchannel)

                val_gen = self.Image_Data_Onechannel_batch(min_x=self.min_x_val, max_x=self.max_x_val,
                                                           min_y=self.min_y_val,
                                                           max_y=self.max_y_val,
                                                           path_LR=rootpath_LR + name,
                                                           path_HR=rootpath_HR + name,
                                                           whichchannel=whichchannel)

                checkpointer = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                               verbose=1,
                                                               monitor='val_loss',
                                                               save_weights_only=False,
                                                               mode='min',
                                                               save_best_only=True,
                                                               factor=0.1,
                                                               patience=2,
                                                               epsilon=1e-4)

                history = model.fit_generator(train_gen,
                                              steps_per_epoch=train_steps,
                                              epochs=2,
                                              validation_data=val_gen,
                                              validation_steps=val_steps,
                                              callbacks=[checkpointer])
                del history
            print(name)

    def Train_continue_batch_rotation(self, path_model_b, path_model_g, path_model_r, rootpath_LR, rootpath_HR):
        '''

        :param path_model_b: b通道模型路径
        :param path_model_g: g通道模型路径
        :param path_model_r: r通道模型路径
        :param rootpath_LR: 低分辨率图像根目录，必须以\\结尾
        :param rootpath_HR: 高分辨率图像根目录，必须以\\结尾
        :return:
        '''
        # 读取文件名
        name_LR = os.listdir(rootpath_LR)
        name_HR = os.listdir(rootpath_HR)

        train_steps = (self.max_x_train - self.min_x_train - 8) * (
                self.max_y_train - self.min_y_train - 8) * 0.25 * 0.25 / 180
        val_steps = (self.max_x_val - self.min_x_val - 8) * (
                self.max_y_val - self.min_y_val - 8) * 0.25 * 0.25 / 180

        for name in name_LR[1:]:
            # 导入已训练模型
            b_model = load_model(path_model_b, compile=False)
            b_model.compile(optimizer='adam', loss=self.my_loss, metrics=metrics.mse)
            g_model = load_model(path_model_g, compile=False)
            g_model.compile(optimizer='adam', loss=self.my_loss, metrics=metrics.mse)
            r_model = load_model(path_model_r, compile=False)
            r_model.compile(optimizer='adam', loss=self.my_loss, metrics=metrics.mse)
            for whichchannel in range(3):
                if whichchannel == 0:
                    model = b_model
                    filepath = 'b_channel.hdf5'
                if whichchannel == 1:
                    model = g_model
                    filepath = 'g_channel.hdf5'
                if whichchannel == 2:
                    model = r_model
                    filepath = 'r_channel.hdf5'

                train_gen = self.Image_Data_Onechannel_batch_rotation(min_x=self.min_x_train + 4,
                                                                      max_x=self.max_x_train - 4,
                                                                      min_y=self.min_y_train + 4,
                                                                      max_y=self.max_y_train - 4,
                                                                      path_LR=rootpath_LR + name,
                                                                      path_HR=rootpath_HR + name,
                                                                      whichchannel=whichchannel)

                val_gen = self.Image_Data_Onechannel_batch_rotation(min_x=self.min_x_val + 4, max_x=self.max_x_val - 4,
                                                                    min_y=self.min_y_val + 4,
                                                                    max_y=self.max_y_val - 4,
                                                                    path_LR=rootpath_LR + name,
                                                                    path_HR=rootpath_HR + name,
                                                                    whichchannel=whichchannel)

                checkpointer = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                               verbose=1,
                                                               monitor='val_loss',
                                                               save_weights_only=False,
                                                               mode='min',
                                                               save_best_only=True,
                                                               factor=0.1,
                                                               patience=2,
                                                               epsilon=1e-4)

                history = model.fit_generator(train_gen,
                                              steps_per_epoch=train_steps,
                                              epochs=1,
                                              validation_data=val_gen,
                                              validation_steps=val_steps,
                                              callbacks=[checkpointer])
                del history
            print(name)

    def Gen_img(self, path_model_b, path_model_g, path_model_r, path_LR, path_save=r'save.bmp'):
        '''
        :param path_model_b: b通道模型路径
        :param path_model_g: g通道模型路径
        :param path_model_r: r通道模型路径
        :param path_LR: 低分辨率图路径
        :param path_save: 生成的高分辨率图保存路径
        :return:
        '''
        b_model = load_model(path_model_b, compile=False)
        b_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
        b_model.summary()
        g_model = load_model(path_model_g, compile=False)
        g_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
        r_model = load_model(path_model_r, compile=False)
        r_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
        Img_LR = cv2.imread(path_LR, flags=1)
        Img_LR = cv2.copyMakeBorder(Img_LR, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        # Img_LR = Img_LR / 255
        b_LR, g_LR, r_LR = cv2.split(Img_LR)
        patch = np.zeros((1, 2, 2, 1))
        GenHR_split = np.zeros((2160, 3840))
        for whichchannel in range(3):
            if whichchannel == 0:
                split = b_LR
                model = b_model
            if whichchannel == 1:
                split = g_LR
                model = g_model
            if whichchannel == 2:
                split = r_LR
                model = r_model
            for i in range(0, 2160, 4):
                for j in range(0, 3840, 4):
                    # time_start = time.time()  # 开始计时
                    patch[0][0][0][0] = split[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                    patch[0][0][1][0] = split[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                    patch[0][1][0][0] = split[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                    patch[0][1][1][0] = split[1 + int(i / 4) + 1][1 + int(j / 4) + 1]
                    pred = model.predict(patch, batch_size=1)
                    conv2d_0 = Model(inputs=model.input, outputs=model.get_layer('conv2d').output)
                    result_conv2d_0 = conv2d_0.predict(patch)[0]
                    conv2d_1 = Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output)
                    result_conv2d_1 = conv2d_1.predict(patch)[0]
                    conv2d_2 = Model(inputs=model.input, outputs=model.get_layer('conv2d_2').output)
                    result_conv2d_2 = conv2d_2.predict(patch)[0]
                    conv2d_3 = Model(inputs=model.input, outputs=model.get_layer('conv2d_3').output)
                    result_conv2d_3 = conv2d_3.predict(patch)[0]
                    conv2d_4 = Model(inputs=model.input, outputs=model.get_layer('conv2d_4').output)
                    result_conv2d_4 = conv2d_4.predict(patch)[0]
                    conv2d_5 = Model(inputs=model.input, outputs=model.get_layer('conv2d_7').output)
                    result_conv2d_5 = conv2d_5.predict(patch)[0]
                    print('hh')
                    '''
                    conv2d_6 = Model(inputs=model.input, outputs=model.get_layer('conv2d_6').output)
                    result_conv2d_6 = conv2d_6.predict(patch)[0]
                    conv2d_7 = Model(inputs=model.input, outputs=model.get_layer('conv2d_7').output)
                    result_conv2d_7 = conv2d_7.predict(patch)[0]
                    conv2d_8 = Model(inputs=model.input, outputs=model.get_layer('conv2d_8').output)
                    result_conv2d_8 = conv2d_8.predict(patch)[0]
                    conv2d_9 = Model(inputs=model.input, outputs=model.get_layer('conv2d_9').output)
                    result_conv2d_9 = conv2d_9.predict(patch)[0]
                    conv2d_10 = Model(inputs=model.input, outputs=model.get_layer('conv2d_10').output)
                    result_conv2d_10 = conv2d_10.predict(patch)[0]
                    dense_0 = Model(inputs=model.input, outputs=model.get_layer('dense').output)
                    result_dense_0 = dense_0.predict(patch)[0]
                    weight_Dense_1,bias_Dense_1=model.get_layer('dense').get_weights()
                    '''

                    pass
                    for m in range(4):
                        for n in range(4):
                            if pred[0][0][0][m * 4 + n] > 1:
                                pred[0][0][0][m * 4 + n] = 1
                            print(pred[0][0][0][m * 4 + n])
                            GenHR_split[i + m][j + n] = pred[0][0][0][m * 4 + n]
                    # time_end = time.time()  # 结束计时
                    # time_c = time_end - time_start  # 运行所花时间
                    # print('time cost', time_c, 's')
                    # pass
            if whichchannel == 0:
                Gen_b = GenHR_split
            if whichchannel == 1:
                Gen_g = GenHR_split
            if whichchannel == 2:
                Gen_r = GenHR_split
            print(whichchannel)
        GenHR_Img = cv2.merge([Gen_b, Gen_g, Gen_r])  # 合并
        cv2.imwrite(path_save, GenHR_Img)

    def Gen_img_batch(self, path_model_b, path_model_g, path_model_r, path_LR, path_save=r'save.bmp'):
        '''
        :param path_model_b: b通道模型路径
        :param path_model_g: g通道模型路径
        :param path_model_r: r通道模型路径
        :param path_LR: 低分辨率图路径
        :param path_save: 生成的高分辨率图保存路径
        :return:
        '''
        batch_size = 1728
        b_model = load_model(path_model_b, compile=False)
        b_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
        g_model = load_model(path_model_g, compile=False)
        g_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
        r_model = load_model(path_model_r, compile=False)
        r_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
        Img_LR = cv2.imread(path_LR, flags=1)
        Img_LR = cv2.copyMakeBorder(Img_LR, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        # Img_LR = Img_LR / 255
        b_LR, g_LR, r_LR = cv2.split(Img_LR)
        patch = np.zeros((batch_size, 2, 2, 1))
        GenHR_split = np.zeros((2160, 3840))
        location = np.zeros((batch_size, 2))
        for whichchannel in range(3):
            if whichchannel == 0:
                split = b_LR
                model = b_model
            if whichchannel == 1:
                split = g_LR
                model = g_model
            if whichchannel == 2:
                split = r_LR
                model = r_model
            num = 0
            for i in range(0, 2160, 4):
                print(i)
                # time_start = time.time()  # 开始计时
                for j in range(0, 3840, 4):
                    # print(j)
                    # time_start = time.time()  # 开始计时
                    if num < batch_size:
                        num = num + 1
                        patch[num - 1][0][0] = split[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[num - 1][0][1] = split[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                        patch[num - 1][1][0] = split[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                        patch[num - 1][1][1] = split[1 + int(i / 4) + 1][1 + int(j / 4) + 1]
                        location[num - 1][0] = i
                        location[num - 1][1] = j
                    else:
                        # print(num)
                        num = 1
                        pred = model.predict(patch, batch_size=batch_size)
                        for k in range(batch_size):
                            for m in range(4):
                                for n in range(4):
                                    if pred[k][0][0][m * 4 + n] > 255:
                                        pred[k][0][0][m * 4 + n] = 255
                                    GenHR_split[int(location[k][0] + m)][int(location[k][1] + n)] = int(
                                        pred[k][0][0][m * 4 + n])
                        patch[num - 1][0][0] = split[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[num - 1][0][1] = split[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                        patch[num - 1][1][0] = split[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                        patch[num - 1][1][1] = split[1 + int(i / 4) + 1][1 + int(j / 4) + 1]
                        location[num - 1][0] = i
                        location[num - 1][1] = j
                # time_end = time.time()  # 结束计时
                # time_c = time_end - time_start  # 运行所花时间
                # print('time cost', time_c, 's')
            if whichchannel == 0:
                Gen_b = copy.deepcopy(GenHR_split)
            if whichchannel == 1:
                Gen_g = copy.deepcopy(GenHR_split)
            if whichchannel == 2:
                Gen_r = copy.deepcopy(GenHR_split)
            '''
            cv2.namedWindow("cs")  # 创建一个窗口，名称cs
            cv2.imshow("cs", GenHR_split)  # 在窗口cs中显示图片
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
            print('Channel is %d' % whichchannel)
        GenHR_Img = cv2.merge([Gen_b, Gen_g, Gen_r])  # 合并
        cv2.imwrite(path_save, GenHR_Img)

    def Gen_img_generator(self, path_model_b, path_model_g, path_model_r, path_LR, path_save=r'save.bmp'):
        '''
        :param path_model_b: b通道模型路径
        :param path_model_g: g通道模型路径
        :param path_model_r: r通道模型路径
        :param path_LR: 低分辨率图路径
        :param path_save: 生成的高分辨率图保存路径
        :return:
        '''
        batch_size = 256
        b_model = load_model(path_model_b, compile=False)
        b_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
        g_model = load_model(path_model_g, compile=False)
        g_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
        r_model = load_model(path_model_r, compile=False)
        r_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
        Img_LR = cv2.imread(path_LR, flags=1)
        Img_LR = cv2.copyMakeBorder(Img_LR, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        Img_LR = Img_LR / 255
        b_LR, g_LR, r_LR = cv2.split(Img_LR)
        patch = np.zeros((batch_size, 2, 2, 1))
        GenHR_split = np.zeros((2160, 3840))
        location = np.zeros((batch_size, 2))
        for whichchannel in range(3):
            if whichchannel == 0:
                split = b_LR
                model = b_model
            if whichchannel == 1:
                split = g_LR
                model = g_model
            if whichchannel == 2:
                split = r_LR
                model = r_model
            num = 0
            for i in range(0, 2160, 4):
                print(i)
                for j in range(0, 3840, 4):
                    # print(j)
                    # time_start = time.time()  # 开始计时
                    if num < batch_size:
                        num = num + 1
                        patch[num - 1][0][0] = split[1 + int(i / 4) + 0][1 + int(j / 4) + 0]
                        patch[num - 1][0][1] = split[1 + int(i / 4) + 0][1 + int(j / 4) + 1]
                        patch[num - 1][1][0] = split[1 + int(i / 4) + 1][1 + int(j / 4) + 0]
                        patch[num - 1][1][1] = split[1 + int(i / 4) + 1][1 + int(j / 4) + 1]
                        location[num - 1][0] = i
                        location[num - 1][1] = j
                    else:
                        yield patch
            if whichchannel == 0:
                Gen_b = GenHR_split
            if whichchannel == 1:
                Gen_g = GenHR_split
            if whichchannel == 2:
                Gen_r = GenHR_split
            print('Channel is %d' % whichchannel)


def InitialTrain():  #
    h = SR_LUT_CNN()
    h.Train(path_LR=r'jingjiawei\downscaled\0.bmp',
            path_HR=r'jingjiawei\GT\0.bmp', whichchannel=0)
    # whichchannel = 0-----B通道
    # whichchannel = 1-----G通道
    # whichchannel = 2-----R通道
    hh = SR_LUT_CNN()
    hh.Train(path_LR=r'jingjiawei\downscaled\0.bmp',
             path_HR=r'jingjiawei\GT\0.bmp', whichchannel=1)
    hhh = SR_LUT_CNN()
    hhh.Train(path_LR=r'jingjiawei\downscaled\0.bmp',
              path_HR=r'jingjiawei\GT\0.bmp', whichchannel=2)


def continue_Train():
    hahaha = SR_LUT_CNN()
    hahaha.Train_continue(path_model_b=r'b_channel.hdf5', path_model_g=r'g_channel.hdf5',
                          path_model_r=r'r_channel.hdf5', rootpath_LR='jingjiawei\\downscaled\\',
                          rootpath_HR='jingjiawei\\GT\\')


def continue_Train_batch():
    hahaha = SR_LUT_CNN()
    hahaha.Train_continue_batch(path_model_b=r'b_channel.hdf5', path_model_g=r'g_channel.hdf5',
                                path_model_r=r'r_channel.hdf5', rootpath_LR='jingjiawei\\downscaled\\',
                                rootpath_HR='jingjiawei\\GT\\')


def continue_Train_batch_rotation():
    hahaha = SR_LUT_CNN()
    hahaha.Train_continue_batch_rotation(path_model_b=r'b_channel.hdf5', path_model_g=r'g_channel.hdf5',
                                         path_model_r=r'r_channel.hdf5', rootpath_LR='jingjiawei\\downscaled\\',
                                         rootpath_HR='jingjiawei\\GT\\')


def predict_generator():
    path_model_b = r'b_channel.hdf5'
    path_model_g = r'b_channel.hdf5'
    path_model_r = r'b_channel.hdf5'
    b_model = load_model(path_model_b, compile=False)
    b_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
    g_model = load_model(path_model_g, compile=False)
    g_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
    r_model = load_model(path_model_r, compile=False)
    r_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metrics.mse)
    h = SR_LUT_CNN()
    test_gen = h.Gen_img_generator(path_model_b=r'b_channel.hdf5', path_model_g=r'b_channel.hdf5',
                                   path_model_r=r'b_channel.hdf5',
                                   path_LR=r'jingjiawei\downscaled\0.bmp')
    pred = b_model.predict_generator(generator=test_gen, steps=960 * 540 / 256)
    print(pred)
    pass


# predict_generator()
# model = load_model('b_channel.hdf5')
# layer1 = model.get_layer(index=18)
# weights = layer1.get_weights()
# pass
# Img_LR = cv2.imread('save.bmp', flags=1)
# Img_LR = cv2.copyMakeBorder(Img_LR, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
# Img_LR = Img_LR / 255
# b_LR, g_LR, r_LR = cv2.split(Img_LR)
# pass
def InitialTrain_batch():  #
    h = SR_LUT_CNN()
    h.Train_batch(path_LR=r'jingjiawei\downscaled\0.bmp',
                  path_HR=r'jingjiawei\GT\0.bmp', whichchannel=0)
    # whichchannel = 0-----B通道
    # whichchannel = 1-----G通道
    # whichchannel = 2-----R通道
    hh = SR_LUT_CNN()
    hh.Train_batch(path_LR=r'jingjiawei\downscaled\0.bmp',
                   path_HR=r'jingjiawei\GT\0.bmp', whichchannel=1)
    hhh = SR_LUT_CNN()
    hhh.Train_batch(path_LR=r'jingjiawei\downscaled\0.bmp',
                    path_HR=r'jingjiawei\GT\0.bmp', whichchannel=2)


def InitialTrain_batch_rotation():  #
    h = SR_LUT_CNN()
    h.Train_batch_rotation(path_LR=r'jingjiawei\downscaled\0.bmp',
                           path_HR=r'jingjiawei\GT\0.bmp', whichchannel=0)
    # whichchannel = 0-----B通道
    # whichchannel = 1-----G通道
    # whichchannel = 2-----R通道
    hh = SR_LUT_CNN()
    hh.Train_batch_rotation(path_LR=r'jingjiawei\downscaled\0.bmp',
                            path_HR=r'jingjiawei\GT\0.bmp', whichchannel=1)
    hhh = SR_LUT_CNN()
    hhh.Train_batch_rotation(path_LR=r'jingjiawei\downscaled\0.bmp',
                             path_HR=r'jingjiawei\GT\0.bmp', whichchannel=2)


# InitialTrain_batch()
# InitialTrain()

# continue_Train_batch()
# continue_Train()

h = SR_LUT_CNN()
h.Gen_img_batch(path_model_b=r'b_channel.hdf5', path_model_g=r'g_channel.hdf5', path_model_r=r'r_channel.hdf5',path_LR=r'jingjiawei\downscaled\23.bmp', path_save=r'save23.bmp')

#InitialTrain_batch_rotation()
#continue_Train_batch_rotation()