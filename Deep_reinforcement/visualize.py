import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from variable_setup import *
import time
from function import *
import random

def plot_graph(df_path, mode='scores', show_step=550, batch_size=16):
    if mode == 'scores':
        df = pd.read_csv(df_path)
        df['game'] = range(len(df))
       
        df = df[(df >= 49)] #cut score from train step
        df = df[df['game'] <= show_step] 
        df.plot.line(x='game', y='scores')
        plt.xlabel('game')
        plt.ylabel('scores')
        plt.show()
        print(max(df['scores']))

    elif mode == 'loss':
        df = pd.read_csv(df_path)
        df['game'] = range(batch_size, batch_size*(len(df)+1), batch_size)
        df = df.iloc[range(int(show_step/batch_size))] 
        df.plot.scatter(x='game', y='loss')
        plt.show()

def analyze_value(sample, model):
    store_history = load_obj('store_history')
    print('store_history sample:{}'.format(len(store_history)))
    random_sample = random.sample(store_history, sample)
    for j in range(sample):
        x_t = random_sample[j][0]
        print('sample at scores:{}'.format(random_sample[j][1]))
        rows = 2
        columns = 2
        fig = plt.figure(figsize=(7, 7))
        for i in range(1, img_channels+1):
            fig.add_subplot(rows, columns, i)
            img = x_t[:, :, i-1]
            plt.imshow(img, cmap='gray')
            time.sleep(0.2)
    
        s_t = x_t.reshape(1, img_rows, img_cols, img_channels)
        q = model.predict(s_t, verbose=0)
        fig = plt.figure(figsize=(7, 7))
        plt.bar(['Do nothing'], q[0][0], color ='skyblue')
        plt.bar(['Jump'], q[0][1], color ='salmon')
        plt.ylabel('Action Value')
        plt.show()
        print(q)

    