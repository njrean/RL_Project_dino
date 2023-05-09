class DinoAgent:
    def __init__(self, game): #takes game as input for taking actions
        self._game = game; 
        self.jump(); #to start the game, we need to jump once
    
    #check playing status
    def is_running(self):
        return self._game.get_playing()
    
    #check crashed status (is game over)
    def is_crashed(self):
        return self._game.get_crashed()
    
    #action jump
    def jump(self):
        self._game.press_up()

    #action crouch down
    # def duck(self):
    #     self._game.press_down()