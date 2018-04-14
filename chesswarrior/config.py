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


class ModelConfig(object):
    """Model Configuration"""

    cnn_filter_num = 256
    res_later_num = 7
    # TODO add your keras structure
    pass

class TrainerConfig(object):
    """Training Configuration"""
    batch_size = 4
    learning_rate = 0.02
    l2_reg = 1e-4
    epoches = 100

    save_interval = 20
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
