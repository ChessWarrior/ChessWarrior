"""ChessWarrior play chess"""

import logging
import os
import matplotlib.pyplot as plt
import time
import multiprocessing

import numpy as np
import keras
from keras.models import load_model
import chess 

from .model import ChessModel
from .config import Config
from .utils import convert_board_to_plane,  get_all_possible_moves, first_person_view_fen, evaluate_board


logger = logging.getLogger(__name__)


class Player(object):
    """Using the best model to play"""

    # Not urgent
    def __init__(self, config: Config):
        self.config = config
        self.model_path = self.config.resources.best_model_dir
        self.model = None
        self.board = None
        self.choise = None
        self.search_depth = 5
        self.move_value = {}
        self.move_hash = {}
        self.policies = []
        self.cnt = []
        self.INF = 0x3f3f3f3f

    def start(self, choise):
        try:
            self.model = load_model(os.path.join(self.config.resources.best_model_dir, "best_model.h5"))
            logger.info("Load best model successfully.")
        except OSError:
            logger.fatal("Model cannot find!")
            return

        self.board = chess.Board()
        self.choise = choise
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
            start = time.time()
            my_move = self.play()   #get ai move
            end = time.time()
            print("time: %.2f" % (end - start))
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
        policy, value = self.model.predict(feature_plane, batch_size=1)
        print("value: %.2f" % value[0][0])
        print("policy: %.2f" % policy[0][0])
        #Attention policy is like [[0,0,1,......]]
        legal_moves = self.board.legal_moves
        
        '''
        return alpha_beta_search()
        '''

        self.alpha_beta_search(self.board, self.search_depth, -self.INF, self.INF)
        candidates = {}  # {move : policy}
        for move in legal_moves:
            if move in self.move_value:
                v = self.move_value[move]
                move = move.uci()
                k = self.move_hash[move]
                p = policy[0][k]
                candidates[move] = p*(1+v)
        
        x =  sorted(candidates.items(), key=lambda x:x[1], reverse=True)
        
        print("evaluation: %f" % x[0][1])
        self.policies.append(x[0][1])

        #plt.plot(range(len(self.policies)), self.policies)
        #plt.show()

        return x[0][0]

    def self_play(self):
        pass

    def alpha_beta_search(self, board, depth, alpha, beta):
        if depth == 0:
            return self.valuation(board)
        if self.search_depth > depth:
            board = chess.Board(first_person_view_fen(board.fen(),1))

        legal_moves_list = []
        feature_plane = convert_board_to_plane(board.fen())
        feature_plane = feature_plane[np.newaxis, :]
        policy_arr, value = self.model.predict(feature_plane, batch_size=1)
        policy_arr = policy_arr[0]

        policy_list = []
        for move in board.legal_moves:
            board.push(move)
            legal_moves_list.append(move)
            policy_list.append(policy_arr[self.move_hash[move.uci()]])
            board.pop()

        move_list = []
        policy_sort = sorted(policy_list, reverse=True)
        for i in range(min(4, len(legal_moves_list))):
            move = legal_moves_list[policy_list.index(policy_sort[i])]
            move_list.append(move)


        for move in move_list:
            board.push(move)
            value = -self.alpha_beta_search(board, depth-1, -beta, -alpha)
            board.pop()
            if self.search_depth == depth:
                self.move_value[move] = value
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return alpha
    
    def valuation(self, board):
        weight = 0.2
        feature_plane = convert_board_to_plane(board.fen())
        feature_plane = feature_plane[np.newaxis, :]
        _, value = self.model.predict(feature_plane, batch_size=1)
        return value[0][0] * weight + evaluate_board(board.fen()) * (1 - weight)


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
