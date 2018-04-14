"""ChessWarrior data processing"""

import os
import logging
import json
from queue import Queue

from .config import Config
from .utils import *


logger = logging.getLogger(__name__)


class DataReader(object):
    """ read pgn files and transform them into json 
    
    """
    def __init__(self, config: Config):
        self.config = config.resources
        self.pgn_filepath = self.config.sl_raw_data_dir
        self.json_filepath = self.config.sl_processed_data_dir
        self.env = ChessBoard()
        self.move_hash = {}
        self.data_buffer = []

    def start(self):
        self.env.reset()
        all_moves = get_all_possible_moves()
        self.move_hash = {move: i for (i, move) in enumerate(all_moves)}
        logger.info("Start to generate data.")
        self.read()

    def read(self):
        """read pgn files and save as json
        data format: class Data
        """
        file_queue = Queue()
        files = os.listdir(self.pgn_filepath)
        for file in files:
            file_queue.put(file)
        n_counter = 0
        while not file_queue.empty():
            filename = file_queue.get()
            logger.info("read %s" % filename)
            with open(self.pgn_filepath+"\\"+filename, 'r') as pgn_file:
                # TODO your read code here

                # Note: about 1000kb~5000kb save as one json file to reduce memory pressure
                name = os.path.splitext(filename)
                if n_counter == 1000:  #n_counter == len(self.data_buffer)
                    with open(self.json_filepath + name[0] + ".json") as file:
                        try:
                            json.dump(self.data_buffer, file)
                            self.data_buffer = []
                        except IOError:
                            logger.fatal("Dump json file failed!")
                            raise
