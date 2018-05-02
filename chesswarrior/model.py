"""ChessWarrior model"""

import logging

import keras

from .config import Config

from keras.models import Model
from keras.layers import Input,Dense, Dropout, Flatten, Activation, Reshape, merge
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta

class ChessModel(object):
    """
    define the structure of neural network
    """
    def __init__(self, config: Config):
        self.config = config
        pass


    def add_rsnet(self, input_data):
        """
        Add one layer residual network to origial network
        ref: https://zhuanlan.zhihu.com/p/21586417
        :param input_data: whole_nn
        :return: new_whole_nn
        """
        model_config = self.config.model
        block1 = Conv2D()(input_data)
        block1 = BatchNormalization()(block1)

        block2 = Conv2D()(block1)
        block2 = BatchNormalization()(block2)

        output_data = merge([input_data, block2], mode="sum")
        output_data = Activation("relu")
        return output_data



    def build(self):
        model_config = Config.model
        input_data = Input(shape=(18, 8, 8))

        block1 = Conv2D(filters=model_config.cnn_filter_num, kernel_size=, padding="same", data_format="channels_first",
                        activation="relu")(input_data)
        block1 = BatchNormalization(axis=1)(block1)

        for _ in range(model_config):
            block1 = self.add_rsnet(block1)

        block2 = Conv2D()(block1)
        block2 = BatchNormalization()(block2)
        block2 = Flatten()(block2)

        fc1 = Dense()(block2)
        fc1 = Dropout()(fc1)

        predict = Dense()(fc1)

        self.model = Model(inputs=input_data, outputs=predict)
