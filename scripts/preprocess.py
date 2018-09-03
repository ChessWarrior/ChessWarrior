"""
data cleaning
"""
import os
import logging
import json
from queue import Queue

import chess.pgn
import chess

from chesswarrior.config import Config

if __name__=='__main__':
    config = Config()
    file_queue = Queue()
    pgn_filepath = config.resources.sl_raw_data_dir
    files = os.listdir(pgn_filepath)

    for file in files:
        file_queue.put(file) #所有raw文件名

    while not file_queue.empty():
        filename = file_queue.get()
        print("read done")
        with open(pgn_filepath+"/"+filename, "r") as pgn_file:
            name = os.path.splitext(filename)
            offsets = list(chess.pgn.scan_offsets(pgn_file))
            print("offsets done")
            game_counter = 0
            games = []
            for offset in offsets:
                pgn_file.seek(offset)
                next_game = chess.pgn.read_game(pgn_file)
                if next_game.headers["Result"] == '1/2-1/2':
                    continue
                if int(next_game.headers["WhiteElo"]) <= 0 or int(next_game.headers["BlackElo"]) <= 0:
                    continue
                game_counter += 1
                print(game_counter)
                games.append(next_game)
                if (game_counter+1) % 50  == 0:
                    print("save %d" % game_counter)
                    games = [str(game) + "\n\n" for game in games]
                    with open(pgn_filepath+'/'+"clean.pgn","a+") as f:
                        f.writelines(games)
                    games = []