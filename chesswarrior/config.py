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

    value_data_dir = os.path.join(_base_data_dir, "tmp")

    json_size = 1024 #moves in a json file

    min_elo = 600.0 #min_elo weight = 0
    max_elo = 2350.0 #max_elo weight = 1

class ModelConfig(object):
    """Model Configuration"""
    cnn_filter_num = 256
    res_layer_num = 7
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
    learning_rate = 0.01
    epoches = 100
    loss_weights = [1.25, 1.0]
    save_interval = 1
    test_interval = 5
    # TODO add your training super params.
    pass

class PlayerConfig(object):
    """Playing Configuration"""
    # Not urgent
    pass

class Config(object):
    """Configurations"""
    CMD = ['train', 'play', 'data']

    cuda_avaliable = True

    resources = ResourceConfig()

    model = ModelConfig()

    training = TrainerConfig()

    playing = PlayerConfig()
