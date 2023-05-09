from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service
from variable_setup import *

class DinoGame_Env:
    def __init__(self, 
                 chromedriver_path=chromedriver_path):

        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        _service_ = Service(chromedriver_path)

        self._driver = webdriver.Chrome(#executable_path = chrome_driver_path,
                                        service=_service_,
                                        options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)

        try:
            self._driver.get('chrome://dino')
            # self._driver.get('https://dino-chrome.com/en')
        except WebDriverException:
            pass
        
        #set not accererate dino speed
        self._driver.execute_script("Runner.config.ACCELERATION=0")

        #set canvas id
        init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
        self._driver.execute_script(init_script)

    #check is game get crashed
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    
    #check is game still playing
    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")
    
    #function restart game to next episode
    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self._driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)

    #can not use Keys.DOWN !!!
    # def press_down(self):
    #     self._driver.find_element(By.TAG_NAME, "body").send_keys(Keys.DOWN)

    #function return score form
    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)
    
    #pause game
    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")
    
    #resume game
    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")
    
    #close window and driver
    def end(self):
        self._driver.close()

