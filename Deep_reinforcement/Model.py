import os
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard
import numpy as np
import time
import random
from collections import deque

import matplotlib.pyplot as plt

from variable_setup import *
from function import *
import math

def buildmodel():
    print("Now we build the model")
    
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same',strides=(4, 4),input_shape=(img_cols,img_rows,img_channels)))  #80*80*4
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4),strides=(2, 2),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3),strides=(1, 1),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))

    adam = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    
    #create model file if not present
    if not os.path.isfile(model_file_path):
        model.save_weights(model_file_path)
        print("-----Build new model-----")
    
    #if there is saved model, load weight to model
    else:
        model.load_weights(model_file_path)
        print("-----Loaded model-----")
    
    return model

def train_model(model, game_state):

    #initial
    action = np.zeros(ACTIONS)
    action[0] = 1 #initial do nothing first
    loss = 0

    loss_df, scores_df, actions_df, q_values_df = create_df()

    D = deque(maxlen=REPLAY_MEMORY)

    x_t, r_0, terminal, score = game_state.get_state(action) # get next step after performing the action
    s_t = x_t.reshape(1, img_rows, img_cols, img_channels)

    initial_state = s_t 

    t = 1
    game_count = 1
    epsilon = EPSILON
    start_time = time.time()
    #train infinity times
    while(True):

        state = 0

        #########explore and collect data in D#########
        if t % BATCH != 0:
            # reset variable
            state = 1
            action_index = 0
            action = np.zeros(ACTIONS)
            
            #do action by epsilon greedy in every frame
            if random.uniform(0, 1.0) <= epsilon:
                action_index = random.randrange(ACTIONS)
                random_state = 1
            else:
                q = model.predict(s_t, verbose=0)
                action_index = np.argmax(q)
                random_state = 0

            action[action_index] = 1

            #epsilon decay
            if epsilon > FINAL_EPSILON:
                epsilon *= 1-EPSILON_DECAY

            x_t1, r_t, terminal, score = game_state.get_state(action)
            s_t1 = x_t1.reshape(1, img_rows, img_cols, img_channels)

            #store history for update weight in model
            D.append((s_t, action_index, r_t, s_t1, terminal))

            s_t = initial_state if terminal else s_t1 #reset game to initial frame if terminate

            actions_df.loc[len(actions_df)] = str(action_index)+' '+str(random_state)+' '+str(epsilon)

            if terminal == True:
                game_count += 1
                scores_df.loc[len(scores_df)] = score

        #########random some history in D to update weight#########
        if t % BATCH == 0 and len(D) > BATCH:

            game_state._game.pause()
            state = 2

            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, img_rows, img_cols, img_channels))
            targets = np.zeros((BATCH, ACTIONS))
            Q_sa = 0

            for i in range(BATCH):
                state_t = minibatch[i][0]    # 4D stack of images
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]   #reward at state_t due to action_t
                state_t1 = minibatch[i][3]   #next state
                terminal = minibatch[i][4]   #wheather the agent died or survided due the action
                
                inputs[i] = state_t

                targets[i] = model.predict(state_t, verbose=0)  # predicted q values
                Q_sa = model.predict(state_t1, verbose=0)      #predict q values for next step

                if terminal:
                    targets[i, action_t] = reward_t # if terminated, only equals reward
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
            
            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_values_df.loc[len(q_values_df)] = np.max(Q_sa)

            #########save model and all scores########
            if t % BATCH*3 == 0:
                model.save_weights('./save_model/model'+str(math.floor(t/100))+'.h5')
                save_df(loss_df, scores_df, actions_df, q_values_df)

            # game_state._game.resume()
            game_count += 1
            scores_df.loc[len(scores_df)] = score

            game_state.obs_img.clear()
            game_state._game.restart()

        t = t + 1

        #########print progess#########
        print('State: {} Step: {} Game: {} Action: {} Random: {} EPSILON: {} spr: {}'.format(state, t, game_count, action, random_state, epsilon, time.time()-start_time), end='\r')
        
        #########control frame rate to run and collect data#########
        if time.time() - start_time < 0.25:
            time.sleep(0.25 - (time.time() - start_time))

        start_time = time.time()

#for behavior of model
def run_train_model(model, game_state, EPSILON=0.05):
    action = np.zeros(ACTIONS)

    store_history = deque(maxlen=1000)

    action[0] = 1 #initial do nothing first

    x_t, r_0, terminal, score = game_state.get_state(action) # get next step after performing the action
    s_t = x_t.reshape(1, img_rows, img_cols, img_channels)

    initial_state = s_t 

    epsilon = EPSILON

    store_history.append((x_t, score))

    start_time = time.time()
    
    while(True):
        
        # reset variable
        action_index = 0
        action = np.zeros(ACTIONS)
        
        #do action by epsilon greedy in every frame
        if random.uniform(0, 1.0) <= epsilon:
            action_index = random.randrange(ACTIONS)
        else:
            q = model.predict(s_t, verbose=0)
            action_index = np.argmax(q)

        action[action_index] = 1

        x_t1, r_t, terminal, score = game_state.get_state(action)
        s_t1 = x_t1.reshape(1, img_rows, img_cols, img_channels)
        s_t = initial_state if terminal else s_t1

        store_history.append((x_t1, score))

        if score >= 57:
            save_obj(store_history, 'store_history')

        if score >= 100:
            save_obj(store_history, 'store_history')
            game_state._game.end()
            break

        print('Store cout: {}'.format(len(store_history)), end='\r')

        if time.time() - start_time < 0.25:
            time.sleep(0.25 - (time.time() - start_time))

        start_time = time.time()
