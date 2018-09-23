# ChessWarrior
Final Project of Spring 2018 Artificial Intelligence at Tongji Univ.

A Chess AI based on Alphago and Alphago Zero

Currently finished supervised learning with alpha-beta search.

A **pytorch** version can be found here [alpha-chess](https://github.com/zhongyuchen/alpha-chess)

# Demo

**ChessWarrior(white) vs. Stockfish(balck; elo 1600)**

![demo](https://github.com/ChessWarrior/ChessWarrior/blob/master/img/demo.gif)

# Usage
There are three modes: data train and play
- data: generate processed training data
- train: the main process to train neural network
- play: use the best model to play chess

> python run.py -mode data

> python run.py -mode train -lr 0.01

> python run.py -mode play --no-cuda

# Requirements
environment windows/linux

python 3.x
- tensorflow-gpu
- keras==2.1.5
- h5py
- numpy
- python-chess

just 
> pip install -r requirements.txt

# Todo
- [ ] Reinforcement Learning
- [ ] MCTS

# Pretrained models
https://pan.baidu.com/s/1C2FVG9vQds-odrfgw5bUGQ 
passwd: imhk
