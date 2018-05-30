"""ChessWarrior utilities"""

import logging
import random
import numpy as np
from functools import reduce

import chess

logger = logging.getLogger(__name__)


# --------------------------------------------
# Data
# --------------------------------------------


class Data(object):
    """data format
    board_state is the board(8x8)  -> input
    policy represents a move stored by a int number because there is a hash between moves and int(see hash) -> output
    value represents a result -> output
    """
    def __init__(self, board_state, policy, value) -> (chess.Board, int, int):
        self.board_state = board_state
        self.policy = policy
        self.value = value


def get_all_possible_moves():
    """return a list of all possible move in the chess"""
    """
    Creates the labels for the universal chess interface into an array and returns them
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                           [(l1, t) for t in range(8)] + \
                           [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                           [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels_array.append(l + '2' + l + '1' + p)
            labels_array.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(l + '2' + l_l + '1' + p)
                labels_array.append(l + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(l + '2' + l_r + '1' + p)
                labels_array.append(l + '7' + l_r + '8' + p)
    return labels_array

# --------------------------------------------
# Environment
# --------------------------------------------

class ChessBoard(object):
    """Board situation"""
    def __init__(self):
        self.board = None
        self.current_player = None
        self.num_halfmoves = 0
        self.winner = None
        self.resigned = False
        self.result = None

    def reset(self):
        """reset to initial state"""
        self.board = chess.Board()
        self.num_halfmoves = 0
        self.winner = None
        self.resigned = False
        self.result = None
        return self

    def step(self, move):
        """take a move; move come from chess.move.uci()"""
        self.board.push_uci(move)
        self.num_halfmoves += 1

PIECES_ORDER = 'KQRBNPkqrbnp'
PIECES_INDEX = {PIECES_ORDER[i] : i for i in range(12)}
EXTEND_SPACE = {
'1' : '1',
'2' : '11',
'3' : '111',
'4' : '1111',
'5' : '11111',
'6' : '111111',
'7' : '1111111',
'8' : '11111111',
'/' : ''
}

labels = get_all_possible_moves()
label_len = len(labels)
flipped_uci_labels = list("".join([str(9 - int(ch)) if ch.isdigit() else ch for ch in chess_move]) \
                          for chess_move in labels)
get_flipped_uci_pos = [flipped_uci_labels.index(chess_move) for chess_move in labels]

def is_black_turn(fen):
    return fen.split(' ')[1] == 'b'

def evaluate_board(fen):
    chess_piece_value = {'Q' : 14, 'R' : 5, 'B' : 3.25, 'K' : 3, 'N' : 3, 'P' : 1}
    current_value = 0.0
    total_value = 0.0
    for ch in fen.split(' ')[0]:
        if not ch.isalpha():
            continue
        if ch.isupper():
            current_value += chess_piece_value[ch]
            total_value += chess_piece_value[ch]
        else:
            current_value -= chess_piece_value[ch.upper()]
            total_value += chess_piece_value[ch.upper()]

    value_rate = current_value / total_value
    if is_black_turn(fen):
        value_rate = -value_rate
    return np.tanh(value_rate * 3)

def get_board_string(board_fen_0):

    rows = board_fen_0.split('/')
    board_string = reduce(lambda x, y: x + y,
                          list(reduce(lambda x, y: x + y, list(map(lambda x: x if x.isalpha() else EXTEND_SPACE[x], row)))
                        for row in rows))
    assert len(board_string) == 64
    return board_string

def get_history_plane(board_fen):

    board_fen_list = board_fen.split(' ')

    history_plane = np.zeros(shape=(12, 8, 8))

    board_string = get_board_string(board_fen_list[0])

    for i in range(8):
        for j in range(8):
            piece = board_string[(i << 3) | j]
            if piece.isalpha():
                history_plane[PIECES_INDEX[piece]][i][j] = 1
    return history_plane


def fen_positon_to_my_position(fen_position):
    return 8 - int(fen_position[1]), ord(fen_position[0]) - ord('a')

def get_auxilary_plane(board_fen):

    board_fen_list = board_fen.split(' ')

    en_passant_state = board_fen_list[3]
    en_passant_plane = np.zeros((8, 8))
    if en_passant_state != '-':
        position = fen_positon_to_my_position(en_passant_state)
        en_passant_plane[position[0]][position[1]] = 1
    fifty_move_count = eval(board_fen_list[4])
    fifty_move_plane = np.full((8, 8), fifty_move_count)

    castling_state = board_fen_list[2]

    K_castling_plane = np.full((8, 8), int('K' in castling_state))
    Q_castling_plane = np.full((8, 8), int('Q' in castling_state))
    k_castling_plane = np.full((8, 8), int('k' in castling_state))
    q_castling_plane = np.full((8, 8), int('q' in castling_state))

    auxilary_plane = np.array([K_castling_plane, Q_castling_plane, k_castling_plane,
                               q_castling_plane, fifty_move_plane, en_passant_plane])

    assert auxilary_plane.shape == (6, 8, 8)
    return auxilary_plane


def get_feature_plane(board_fen):

    history_plane = get_history_plane(board_fen)
    auxilary_plane = get_auxilary_plane(board_fen)
    feature_plane = np.vstack((history_plane, auxilary_plane))
    assert feature_plane.shape == (18, 8, 8)
    return feature_plane

def first_person_view_fen(board_fen, flip):

    if not flip:
        return board_fen

    board_fen_list = board_fen.split(' ')
    rows = board_fen_list[0].split('/')

    rows = [reduce(lambda x, y : x + y, list(map(lambda ch: ch.lower() if ch.isupper() else ch.upper(), row))) for row in rows]
    board_fen_list[0] = '/'.join(reversed(rows))

    board_fen_list[1] = 'w' if board_fen_list[1] == 'b' else 'b'

    board_fen_list[2] = "".join(sorted("".join(ch.lower() if ch.isupper() else ch.upper() for ch in board_fen_list[2])))

    ret_board_fen = ' '.join(board_fen_list)
    return ret_board_fen

def first_person_view_policy(policy, flip):

    if not flip:
        return policy

    return np.array([policy[pos] for pos in get_flipped_uci_pos])


def convert_board_to_plane(board_fen):
    return get_feature_plane(first_person_view_fen(board_fen, is_black_turn(board_fen)))


# --------------------------------------------
# Train
# --------------------------------------------


class Batchgen(object):
    '''generate batches
    data is a list of the class Data
    batch_size is int type
    '''
    def __init__(self, data, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.data = data
        if shuffle:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.batches = [self.standardize(data[i:i + batch_size]) for i in range(0, len(data), batch_size)]

    def standardize(self, data):
        '''
        https://www.xqbase.com/protocol/pgnfen2.htm
        :param data:
        :return:
        '''
        feature_plane_list = []

        value_list = []
        for board_fen, value in data:
            feature_plane = get_feature_plane(board_fen)
            round_time = int(board_fen.split(' ')[5])
            value = float(value)
            learning_value = np.tanh(value)
            
            feature_plane_list.append(feature_plane)
            value_list.append(learning_value)

        return np.array(feature_plane_list, dtype=np.float32), \
               np.array(value_list, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.batches:
            yield batch

        raise StopIteration
