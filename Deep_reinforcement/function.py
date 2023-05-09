import pickle
import numpy as np
import cv2
from collections import deque
from io import BytesIO
import base64
from PIL import Image
from variable_setup import *
import pandas as pd

def save_obj(obj, name):
    with open('objects/'+ name + '.pkl', 'wb') as f: #dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def create_df():
    loss_df = pd.DataFrame(columns =['loss'])
    scores_df = pd.DataFrame(columns = ['scores'])
    actions_df = pd.DataFrame(columns = ['actions'])
    q_values_df = pd.DataFrame(columns = ['qvalues'])

    return loss_df, scores_df, actions_df, q_values_df

def save_df(loss_df, scores_df, actions_df, q_values_df):
    loss_df.to_csv(loss_file_path,index=False)
    scores_df.to_csv(scores_file_path,index=False)
    actions_df.to_csv(actions_file_path,index=False)
    q_values_df.to_csv(q_value_file_path,index=False)

def grab_screen(_driver):
    image_b64 = _driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen) #processing image
    return image

def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #RGB to Grey Scale leave only 1 channel
    _, image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)

    image = image[ :, :450] #Crop
   
    image_border = cv2.copyMakeBorder(
                    image,
                    top=200,
                    bottom=100,
                    left=0,
                    right=0,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
    image_border = cv2.resize(image_border, (240,240))
    return  image_border