"""ChessWarrior data processing"""

import os
import logging
import json
from queue import Queue
import chess.pgn
import chess

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
        self.json_size = self.config.json_size

    def start(self):
        self.env.reset()
        all_moves = get_all_possible_moves()
        self.move_size = int(len(all_moves))
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
            file_queue.put(file) #所有raw文件名
        n_counter = 0 #number of moves in data_buffer
        while not file_queue.empty():
            filename = file_queue.get()
            file_counter = 0
            logger.info(f"read {filename}")
            with open(self.pgn_filepath+"\\"+filename, "r") as pgn_file:
                # TODO your read code here
                name = os.path.splitext(filename)
                offsets = list(chess.pgn.scan_offsets(pgn_file))
                logger.info(f"{len(offsets)} games in {filename}")
                game_counter = 0
                for offset in offsets:
                    game_counter+=1
                    pgn_file.seek(offset)
                    next_game = chess.pgn.read_game(pgn_file) #read one game
                    n_counter += self.get_moves(next_game) #get steps
                    # Note: about 1000kb~5000kb save as one json file to reduce memory pressure
                    while n_counter >= self.json_size: #n_counter == len(self.data_buffer)
                        file_counter+=1
                        with open(self.json_filepath + "\\" + name[0] + "_" + str(file_counter) + ".json", "w") as file:
                            n_counter = n_counter - self.json_size
                            try:
                                json.dump(self.data_buffer[0:self.json_size], file) #dump 1024 steps(about 10MB)
                                self.data_buffer = self.data_buffer[self.json_size:]
                            except IOError:
                                logger.fatal("Dump json file failed!")
                                raise
                    if game_counter % 200 == 0:
                        logger.info(f"processed {game_counter} games")
        if len(self.data_buffer) != 0:  #dump last steps
            with open(self.json_filepath + "\\" + "last_steps.json", "w") as file:
                try:
                    json.dump(self.data_buffer, file)
                    self.data_buffer = []
                    logger.info("Clear buffer successfully!")
                except IOError:
                    logger.fatal("Dump json file failed!")
                    raise

    def solve_move(self, board, next_move, weight, result):
        ret_move = []
        ret_move.append(board.fen())
        ret_policy = [0.0 for i in range(self.move_size)]
        k=self.move_hash[next_move]
        ret_policy[k] = weight
        ret_move.append(ret_policy)
        ret_move.append(result)
        return ret_move

    def get_moves(self, next_game: chess.pgn.Game):
        """
        get moves from one game and append to data_buffer
        param pgn.Game next_game: a game
        return int: number of moves in the game
        """
        data_len = 0 #moves processed
        result = self.get_result(next_game.headers["Result"])
        white_weight = self.elo_val(int(next_game.headers["WhiteElo"]))
        black_weight = self.elo_val(int(next_game.headers["BlackElo"]))
        board = chess.Board()
        while not next_game.is_end():
            next_game=next_game.variation(0)
            next_move = next_game.move.uci()
            if(data_len % 2 == 0): #white move
                self.data_buffer.append(self.solve_move(board, next_move, white_weight, result))
            else: #black move
                self.data_buffer.append(self.solve_move(board, next_move, black_weight, -result))
            board.push_uci(next_move)
            data_len += 1
        return data_len
    
    def get_result(self, result):
        """
        return int: white_win: 1; black_win: -1; draw: 0;
        """
        if result == '1-0':
            return 1
        elif result == '0-1':
            return -1
        else:
            return 0

    def elo_val(self,elo):
        weight = (elo - self.config.min_elo) / (self.config.max_elo - self.config.min_elo)#debug
        if weight > 1.0:
            weight = 1.0
        elif weight < 0.0:
            weight = 0.0
        return weight

