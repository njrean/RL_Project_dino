#path variables
game_url = "chrome://dino"
# chrome_driver_path = "../chromedriver_win32"
chromedriver_path = r"C\chromedriver_win32\chromedriver.exe"
#history save path
loss_file_path = "./objects/loss_df.csv"
actions_file_path = "./objects/actions_df.csv"
q_value_file_path = "./objects/q_values.csv"
scores_file_path = "./objects/scores_df.csv"

model_file_path = './save_model/model.h5'

#get image from canvas
getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"

#game parameters
ACTIONS = 2 # possible actions: do 0: nothing, 1: jump, 2: crouch down
GAMMA = 0.99 # decay rate of past observations original 0.99
FINAL_EPSILON = 0.05 # final value of epsilon
EPSILON = 0.4
EPSILON_DECAY = 0.001
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 10000 # number of previous transitions to remember
BATCH = 128 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-3

img_rows , img_cols = 240, 240
img_channels = 4 #We stack 4 frames