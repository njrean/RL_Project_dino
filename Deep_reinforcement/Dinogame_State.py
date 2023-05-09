import os
import pandas as pd
from variable_setup import *
from function import *
import time 

class Dinogame_State:
    def __init__(self, agent, game_env):
        self._agent = agent
        self._game = game_env

        self.obs_img = deque(maxlen=4)
    
    #Do action in game
    def get_state(self, actions):

        score = self._game.get_score() 
        reward = 0.1
        is_over = False #game over

        act = np.argmax(actions)
        
        #do action
        #jump
        if act == 1:
            self._agent.jump()

        #crouch down !!! it not working !!!
        # elif act == 2:
        #     self._agent.duck()

        #other way to collect 4 frame in 1 iteration
        # self.obs_img.clear()
        # for i in range(img_channels):
        #     image = grab_screen(self._game._driver)
        #     self.obs_img.append(image)
        #     time.sleep(0.05)
        # obs = np.stack(self.obs_img, axis=2)

        image = grab_screen(self._game._driver) #have already preprocess in grap_screen function 
        
        self.obs_img.append(image)

        if len(self.obs_img) < 4:
            obs = np.stack([image] * 4, axis=2)
        else:
            obs = np.stack(self.obs_img, axis=2)

        if self._agent.is_crashed():
            self.obs_img.clear()
            self._game.restart()
            reward = -1
            is_over = True

        return obs, reward, is_over, score #return the Experience tuple