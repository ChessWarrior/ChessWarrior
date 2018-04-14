"""ChessWarrior utilities"""


import chess
import logging
import random

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


# --------------------------------------------
# Train
# --------------------------------------------

class Batchgen(object):
    """generate batches
    data is a list of the class Data
    batch_size is int type
    """
    def __init__(self, data, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.data = data
        if shuffle:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.batches:
            yield batch

        raise StopIteration
