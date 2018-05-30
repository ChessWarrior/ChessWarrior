"""ChessWarrior train model"""

import logging
import os
import json
from queue import Queue

import keras
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical

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
                # h5 file (model and weights)
                self.model = load_model(os.path.join(self.config.resources.best_model_dir, "best_model.h5"))
                logger.info('load last trained best model.')
            except OSError:
                self.ChessModel = ChessModel(config=self.config)
                self.ChessModel.build()
                self.model = self.ChessModel.model
                self.model.compile(optimizer=keras.optimizers.adam(lr=self.config.training.learning_rate),
                                   loss='mean_squared_error')
                logger.info('A new model is born.')


        self.data_files = os.listdir(self.config.resources.value_data_dir)
        if not self.data_files:
            logger.fatal("No Porcessed data!")
            raise RuntimeError("No processed data!")

        with open(os.path.join(self.config.resources.best_model_dir, "epoch.txt"), "r") as file:
            self.epoch0 = int(file.read()) + 1
        self.training()

    def training(self):

        for epoch in range(self.epoch0, self.epoch0 + self.config.training.epoches):
            logger.info('epoch %d start!' % epoch)

            for data_file in self.data_files:
                with open(os.path.join(self.config.resources.value_data_dir, data_file), "r", encoding='utf-8') as file:
                    data = json.load(file)
                    batches = Batchgen(data, self.config.training.batch_size)
                    for feature_plane_array, value_array in batches:
                        history_callback = self.model.fit(feature_plane_array, value_array,
                                        validation_split=0.1, shuffle=True, verbose=2)
                        loss = history_callback.history["loss"][0]
                        val_loss =  history_callback.history["val_loss"][0]
                        logger.debug("loss: %f val_loss: %f " % (loss, val_loss))

            self.model.save(os.path.join(self.config.resources.best_model_dir, "best_model.h5"))
            with open(os.path.join(self.config.resources.best_model_dir, "epoch.txt"), "w") as file:
                file.write(str(epoch))


    def f1_score(self):
        # Not urgent
        pass

    def elo(self):
        # Not urgent
        pass
