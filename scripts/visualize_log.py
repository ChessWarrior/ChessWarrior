'''
This file is to plot loss
'''
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import paramiko

def get_log():
    transport = paramiko.Transport(('xxxxxxxxx', 22))
    transport.connect(username='xxxxxx', password='xxxxxx')
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.get('/home/xxxxx/Downloads/ChessWarrior/nohup.out', r"D:\Coding\Pycharm\ChessWarrior2\log.txt")

def regular_expr():
    with open(r"D:\Coding\Pycharm\ChessWarrior2\log.txt", "r") as f:
        data = f.read()
    data = re.findall('loss: \d+\.\d+', data)
    data = [(data[idx].split(' ')[1]+" "+data[idx+1].split(' ')[1]) for idx in range(0,len(data),2)]
    with open(r"D:\Coding\Pycharm\ChessWarrior2\p_log.txt", "w") as f:
        f.write('\n'.join(data))

def visualize():
    df = pd.read_table(r"D:\Coding\Pycharm\ChessWarrior2\p_log.txt",
     names=['loss','val_loss'], delim_whitespace=True)
    #print(df.head())
    plt.figure('0')
    font = {
        'family':'monospace',
        'size':18
    }
    h1,=plt.plot(range(len(df["val_loss"])), df["val_loss"], 'b')
    h2,=plt.plot(range(len(df["loss"])), df["loss"], 'r')
    plt.xlabel('iter',font)
    plt.ylabel('loss',font)
    
    plt.title('Loss',font)
    plt.legend([h1,h2], ['val_loss', 'train_loss'], loc='upper right', prop=font)
    plt.show()
    plt.close('0')
 
if __name__=='__main__':
    
    get_log()
    regular_expr()
    visualize()
