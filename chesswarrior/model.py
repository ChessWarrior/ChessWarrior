"""ChessWarrior model"""

import logging

import keras

from .config import Config

from keras.models import Model
from keras.layers import Input,Dense, Dropout, Flatten, Activation, Reshape, Add
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.regularizers import l2

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
        block1 = Conv2D(filters=model_config.cnn_filter_num, kernel_size=model_config.cnn_filter_size,
                        padding="same", data_format="channels_first",
                        kernel_initializer="random_normal",bias_initializer="zeros", kernel_regularizer=l2(model_config.l2_regularizer)
                        )(input_data)
        block1 = BatchNormalization(axis=1)(block1)
        block1 = Activation("relu")(block1)

        block2 = Conv2D(filters=model_config.cnn_filter_num, kernel_size=model_config.cnn_filter_size,
                        padding="same", data_format="channels_first",
                        kernel_initializer="random_normal",bias_initializer="zeros", kernel_regularizer=l2(model_config.l2_regularizer)
                        )(block1)
        block2 = BatchNormalization(axis=1)(block2)

        output_data = Add()([input_data, block2])
        output_data = Activation("relu")(output_data)
        return output_data


    def build(self):
        model_config = Config.model
        input_data = Input(shape=(18, 8, 8))

        block1 = Conv2D(filters=model_config.cnn_filter_num, kernel_size=model_config.cnn_first_filter_num,
                        kernel_initializer="random_normal",bias_initializer="zeros", padding="same", data_format="channels_first"
                        )(input_data)
        block1 = BatchNormalization(axis=1)(block1)
        block1 = Activation("relu")(block1)

        for _ in range(model_config.res_layer_num):
            block1 = self.add_rsnet(block1)

        block2_value = Conv2D(filters=4, kernel_size=1, data_format="channels_first",
                              kernel_initializer="random_normal",bias_initializer="zeros",
                               kernel_regularizer=l2(model_config.l2_regularizer)
                               )(block1)
        block2_value = BatchNormalization(axis=1)(block2_value)
        block2_value = Activation("relu")(block2_value)
        block2_value = Flatten()(block2_value)

        fc_value = Dense(units=model_config.value_fc_size, kernel_regularizer=l2(model_config.l2_regularizer),
                         activation="relu", kernel_initializer="random_normal", bias_initializer="zeros",)(block2_value)
        value_out = Dense(units=1, kernel_regularizer=l2(model_config.l2_regularizer),
                          kernel_initializer="random_normal", bias_initializer="zeros", activation="tanh", name="value_out")(fc_value)

        self.model = Model(inputs=input_data, outputs= value_out)

'''if __name__ == '__main__':
    chess_model = ChessModel()
    chess_model.build()'''
