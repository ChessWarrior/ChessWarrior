"""ChessWarrior play chess"""

import logging

import keras

from .model import ChessModel
from .config import Config

logger = logging.getLogger(__name__)


class Player(object):
    """Using the best model to play"""

    # Not urgent
    def __init__(self, config: Config):
        pass

    def start(self):
        pass