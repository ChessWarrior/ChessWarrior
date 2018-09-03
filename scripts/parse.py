'''
parse analyzed pgn fils
'''
import os
import re
from io import StringIO
import json

import chess
import chess.pgn
from tqdm import tqdm

def preprocess():
    base_dir = "Codes/python/data"
    files = os.listdir(base_dir)
    files.sort(key=lambda x:int(x[:-4]))
    dataset = []
    for file in tqdm(files):
        with open(os.path.join(base_dir, file), "r" , encoding='utf-8') as f:
            pgn = chess.pgn.scan_headers(f)
            flag = 0
            for offset, header in pgn:
                if not header.get('FEN') and "Crazyhouse" not in header['Event']:
                    flag = 1
                    break
                    
            if flag:
                f.seek(0)
                pgn = f.read()
                if "eval" in pgn:
                    dataset.append(pgn+"\n\n")

    return dataset

def parse(dataset):
    ans = []
    cnt = 0
    for data in tqdm(dataset):
        evals = re.findall('%eval\s?#?-?\d+\.?\d*', data)
        pgn = chess.pgn.read_game(StringIO(data))
        board = pgn.board()
        moves = pgn.main_line()
        moves = len(list(moves))
        
        if moves - 1 == len(evals) or moves == len(evals):
            for i , move in enumerate(pgn.main_line()):
                board.push(move)
                if i == len(evals):
                    break
                eval = re.sub('%eval', '', evals[i])
                eval = eval.strip()
                if "#" in eval:
                    continue
                board_str = board.fen()
                if "[" in board_str:
                    continue
                ans.append({board_str:eval})
    return ans
    
def clean():
    with open(r"D:\Coding\Pycharm\ChessWarrior\data\value\ans2.txt", "r" ,encoding='utf-8') as f:
        data = json.load(f)
    new_data = []
    cnt = 0
    for d in data:
        if (cnt+1) % 1000 == 0:
            new_data = json.dumps(new_data)
            with open(r"D:\Coding\Pycharm\ChessWarrior\data\value\res" + str((cnt+1) // 1000)+".txt", "w" ,encoding='utf-8') as f:
                f.write(new_data)
            new_data = []
        new_data.append(d)
        cnt += 1
    
    
if __name__=='__main__':
    '''
    data = preprocess()
    print(len(data))
    ans = parse(data)
    ans = json.dumps(ans)
    with open("ans.txt", "w" ,encoding='utf-8') as f:
        f.writelines(ans)
            '''
    clean()

    
    



