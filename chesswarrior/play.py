"""ChessWarrior play chess"""

import logging
import os
import matplotlib.pyplot as plt

import numpy as np
import keras
from keras.models import load_model
import chess 

from .model import ChessModel
from .config import Config
from .utils import convert_board_to_plane,  get_all_possible_moves, first_person_view_fen


logger = logging.getLogger(__name__)


class Player(object):
    """Using the best model to play"""

    # Not urgent
    def __init__(self, config: Config):
        self.config = config
        self.model_path = self.config.resources.best_model_dir
        self.model = None
        self.board = None
        self.move_hash = {}
        self.policies = []
        self.cnt = []

    def start(self, choise):
        try:
            self.model = load_model(os.path.join(self.config.resources.best_model_dir, "best_model.h5"))
            logger.info("Load best model successfully.")
        except OSError:
            logger.fatal("Model cannot find!")
            return

        self.board = chess.Board()

        all_moves = get_all_possible_moves()
        self.move_hash = {move: i for (i, move) in enumerate(all_moves)}

        if choise == 1:
            self.board = chess.Board(first_person_view_fen(self.board.fen(), 1))
            print(self.board)
            while True:
                opponent_move = input("your move:")
                try:
                    convert_move = convert_black_uci(opponent_move)
                    convert_move = self.board.parse_uci(convert_move)
                    break
                except ValueError:
                    logger.info("Opponent make a illegal move")
            logger.info("Your move: %s" % opponent_move)
            self.board.push(convert_move) # other move.  update board
        
        while not self.board.is_game_over():
            my_move = self.play()   #get ai move
            #my_move = convert_black_uci(my_move)
            convert_move = self.board.parse_uci(my_move)
            self.board.push(convert_move) # ai move. update board
            if choise == 1:
                logger.info("AI move: %s" % convert_black_uci(my_move))
            else:
                logger.info("AI move: %s" % my_move)
            
            print(self.board)
            while True:
                opponent_move = input("your move:")
                if opponent_move == "undo": #undo 
                    self.board.pop()
                    self.board.pop()
                    logger.info("Undo done.")
                    continue
                try:
                    if choise == 1:
                        convert_move = convert_black_uci(opponent_move)
                    else:
                        convert_move = opponent_move
                    convert_move = self.board.parse_uci(convert_move)
                    break
                except ValueError:
                    logger.info("Opponent make a illegal move")
            logger.info("Your move: %s" % opponent_move)
            
            self.board.push(convert_move) # other move.  update board

        
        
    
    def play(self):
        """
        return my move
        """
        feature_plane = convert_board_to_plane(self.board.fen())
        feature_plane = feature_plane[np.newaxis, :]
        [policy, value] = self.model.predict(feature_plane, batch_size=1)
        #Attention policy is like [[0,0,1,......]]
        legal_moves = self.board.legal_moves
        
        '''
        return alpha_beta_search()
        '''

        candidates = {}  # {move : policy}
        for move in legal_moves:
            move = move.uci()
            k = self.move_hash[move]
            p = policy[0][k]
            candidates[move] = p
        
        x =  sorted(candidates.items(), key=lambda x:x[1], reverse=True)
        
        print("policy: %f" % x[0][1])
        self.policies.append(x[0][1])

        plt.plot(range(len(self.policies)), self.policies)
        plt.show()

        return x[0][0]

    def self_play(self):
        pass

    def alpha_beta_search(self):
        #TODO: your a-b search code here
        pass


def convert_black_uci(move):
    new_move = []
    for s in move:
        if s.isdigit():
            new_move.append(str(9 - int(s)))
        else:
            new_move.append(s)
    return ''.join(new_move)

def count_piece(board_fen):
    '''
    count many pieces left on the board
    '''
    cnt = 0
    for pos in board_fen:
        if pos.isalpha():
            cnt += 1
    return cnt
