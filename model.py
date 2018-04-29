"""ChessWarrior model"""

import logging

import keras

from .config import Config

class ChessModel(object):
    """
    define the structure of neural network
    """
    def __init__(self, config: Config):
        self.config = config
        pass

    def build(self):
        #TODO: your keras code here
        pass
