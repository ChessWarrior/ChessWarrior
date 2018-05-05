"""ChessWarrior train model"""

import logging
import os
import json
from queue import Queue

import keras
from keras.models import load_model

from .model import ChessModel
from .config import Config
from .utils import Batchgen


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.data_files = None
        self.batches = None
        self.epoch0 = 0
        self.loss = 0

    def start(self):
        if not self.model:
            try:
                self.model = load_model(self.config.resources.best_model_dir+"best_model.h5") #h5 file (model and weights)
            except OSError:
                self.ChessModel = ChessModel(config=self.config)
                self.ChessModel.build()
                self.model = self.ChessModel.model
                self.model.compile(optimizer=keras.optimizers.adagrad(), loss=['categorical_crossentropy', 'mean_squared_error'],
                                   loss_weights=self.config.training.loss_weights, metrics=['accuracy', 'mse'])


        self.data_files = os.listdir(self.config.resources.sl_processed_data_dir)
        if not self.data_files:
            logger.fatal("No Porcessed data!")
            raise RuntimeError("No processed data!")

        with open(self.config.resources.best_model_dir+"\\epoch.txt", "r") as file:
            self.epoch0 = int(file.read())
        self.training()

    def training(self):

        for epoch in range(self.epoch0, self.epoch0 + self.config.training.epoches):
            logger.info('epoch %d start!' % epoch)

            for data_file in self.data_files:
                with open(self.config.resources.sl_processed_data_dir+"\\"+data_file, "r", encoding='utf-8') as file:
                    data = json.load(file)
                    batches = Batchgen(data, self.config.training.batch_size)
                    for batch_data_i in batches:
                        feature_plane_array, policy_array, value_array = batch_data_i
                        self.model.fit(feature_plane_array, [policy_array, value_array], verbose=1, validation_split=0.05, shuffle=True)

            if epoch == self.config.training.test_interval:
                pass
                #self.testing()
            elif epoch == self.config.training.save_interval:
                self.model.save(self.config.resources.best_model_dir+"best_model.h5")
                with open(self.config.resources.best_model_dir+"epoch.txt", "w") as file:
                    file.write(epoch)
            else:
                continue

    def f1_score(self):
        # Not urgent
        pass

    def elo(self):
        # Not urgent
        pass
