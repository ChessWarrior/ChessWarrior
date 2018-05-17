"""ChessWarrior configuration"""

import os

# -----------------------------------
# configurations about some PARAMETERS
# -----------------------------------

class ResourceConfig(object):
    """resouces configuratioins"""
    cur_dir = os.path.abspath(__file__)

    d = os.path.dirname
    base_dir = d(d(cur_dir))

    _base_data_dir = os.path.join(base_dir, "data")

    best_model_dir = os.path.join(_base_data_dir, "model")

    _sl_base_data_dir = os.path.join(_base_data_dir, "human_expert")

    sl_raw_data_dir = os.path.join(_sl_base_data_dir, "raw") #pgn files

    sl_processed_data_dir = os.path.join(_sl_base_data_dir, "processed") #json files

    json_size = 1024 #moves in a json file

    min_elo = 600.0 #min_elo weight = 0
    max_elo = 2350.0 #max_elo weight = 1

class ModelConfig(object):
    """Model Configuration"""
    cnn_filter_num = 256
    res_layer_num = 19
    cnn_first_filter_num = 5
    cnn_filter_size = 3
    l2_regularizer = 1e-5
    value_fc_size = 256
    drop_out_rate = 0.5
    # TODO add your keras structure
    pass

class TrainerConfig(object):
    """Training Configuration"""
    batch_size = 512
    learning_rate = 0.02
    epoches = 30
    loss_weights = [1.25, 1.0]
    save_interval = 1
    test_interval = 5
    # TODO add your training super params.
    pass

class PlayerConfig(object):
    """Playing Configuration"""
    # Not urgent
    pass


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

class Config(object):
    """Configurations"""
    CMD = ['train', 'play', 'data']

    cuda_avaliable = True

    resources = ResourceConfig()

    model = ModelConfig()

    training = TrainerConfig()

    playing = PlayerConfig()

    labels = get_all_possible_moves()

    label_len = len(labels)

    flipped_uci_labels = list("".join([str(9 - int(ch)) if ch.isdigit() else ch for ch in chess_move]) \
                          for chess_move in labels)

    get_flipped_uci_pos = None

Config.get_flipped_uci_pos = [Config.flipped_uci_labels.index(chess_move) for chess_move in Config.labels]