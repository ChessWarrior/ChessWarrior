import argparse
import time


from chesswarrior.config import Config

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    config = Config()
    parser.add_argument('-ch', type=int, help="choise for white or black",default=0)
    args = parser.parse_args()
    input() #忽略
    #都是str类型
    pre_ai_move = None 

    if args.ch == 1: #黑棋，等待对方
        opponent_move = input() #这是str类型
        with open(config.playing.oppo_move_dir, "w") as f:
            f.write(opponent_move) 

    while True:
        #忙轮询 0.1s一次
        while True:
            #ai move
            with open(config.playing.ai_move_dir, "r") as f:
                ai_move = f.read()
            if ai_move == pre_ai_move or not ai_move:
                time.sleep(0.1)
            else:
                print(ai_move)
                pre_ai_move = ai_move
                break
            
        #oppo move
        opponent_move = input()
        with open(config.playing.oppo_move_dir, "w") as f:
            f.write(opponent_move)
