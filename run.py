"""ChessWarrior runner"""

import argparse
import os
import logging

from chesswarrior.config import Config
from chesswarrior.data import DataReader
from chesswarrior.train import Trainer
from chesswarrior.play import Player

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(fmt)
handler = logging.FileHandler('log.txt')
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.addHandler(console)

config = Config()
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help="which mode", choices=config.CMD)
parser.add_argument('--no-cuda', action='store_false', help="Running without cuda")
parser.add_argument('-epoch', type=int, help="training epoches",default=config.training.epoches)
parser.add_argument('-lr', type=float, help="learning rate",default=config.training.learning_rate)
parser.add_argument('-batch_size', type=int, help="batch size",default=config.training.batch_size)
parser.add_argument('-l2_reg', type=float, help="l2_regulation",default=config.model.l2_regularizer)
parser.add_argument('-ch', type=int, help="choise for white or black",default=0)

args = parser.parse_args()


config.cuda_avaliable = args.no_cuda
if not config.cuda_avaliable:

    logger.info('CPU is running')
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    logger.info('GPU is running')

config.training.epoches = args.epoch
config.training.learning_rate = args.lr
config.training.batch_size = args.batch_size
config.training.l2_reg = args.l2_reg

if args.mode == 'data':
    logger.info('Initializing DataReader')
    datareader = DataReader(config)
    datareader.start()
elif args.mode == 'train':
    logger.info('Initializing Trainer')
    trainer = Trainer(config)
    trainer.start()
elif args.mode == 'play':
    logger.info('Initializing Player')
    player = Player(config)
    player.start(args.ch)
else:
    raise RuntimeError("Mode %s is undefined." % args.mode)
