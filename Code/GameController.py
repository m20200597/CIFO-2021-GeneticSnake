# -*- coding: utf-8 -*-
"""
Function which implements the movement of the snake for each iteration
using a Neural Network.
"""
from NN import *

def playSnake(SNAKE,weights):
    while not SNAKE.game_over:
        # extract the game status
        overview = SNAKE.ReturnOverview()
        # transform in numpy input
        netInput = np.array(overview).reshape(-1,inputUnits)
        # Perform the Feed Forward Algorithm with the current status of the game
        # plus the weights of the neural network for this child.
        movesprob = forward_propagation(netInput,weights)
        nextMove = np.argmax(movesprob[0])
        
        # Gets the last direction of movement by the Snake
        if len(SNAKE.moves) > 0:
            if SNAKE.moves[-1] == "LEFT":
                dirLEFT = 1
                dirRIGHT = 0
                dirUP = 0
                dirDOWN = 0
            elif SNAKE.moves[-1] == "RIGHT":
                dirLEFT = 0
                dirRIGHT = 1
                dirUP = 0
                dirDOWN = 0
            elif SNAKE.moves[-1] == "UP":
                dirLEFT = 0
                dirRIGHT = 0
                dirUP = 1
                dirDOWN = 0
            elif SNAKE.moves[-1] == "DOWN":
                dirLEFT = 0
                dirRIGHT = 0
                dirUP = 0
                dirDOWN = 1
        else:
            dirLEFT = 0
            dirRIGHT = 0
            dirUP = 0
            dirDOWN = 0
        
        '''
            
        Based on the previous movement, our snake will either:
            . Continue movement, which will execute the last movement.
            . Move 90 degrees left, which will depend on the last movement to be performed.
            . Move 90 degrees right, which will depend on the last movement to be performed.
        
        '''
        if nextMove == 0: # CONTINUE PREVIOUS MOVEMENT
            SNAKE.LastPredictMovement = 0
            if dirUP == 1:
                SNAKE.Movement_UP()
            elif dirDOWN == 1:
                SNAKE.Movement_DOWN()
            elif dirLEFT == 1:
                SNAKE.Movement_LEFT()
            elif dirRIGHT == 1:
                SNAKE.Movement_RIGHT()
            else:
                SNAKE.Movement_UP()
                
        if nextMove == 1: # 90 DEGREES LEFT
            SNAKE.LastPredictMovement = 1
            if dirUP == 1:
                SNAKE.Movement_LEFT()
            elif dirDOWN == 1:
                SNAKE.Movement_RIGHT()
            elif dirLEFT == 1:
                SNAKE.Movement_DOWN()
            elif dirRIGHT == 1:
                SNAKE.Movement_UP()
            else:
                SNAKE.Movement_LEFT()
                
        if nextMove == 2: # 90 DEGREES RIGHT
            SNAKE.LastPredictMovement = 2
            if dirUP == 1:
                SNAKE.Movement_RIGHT()
            elif dirDOWN == 1:
                SNAKE.Movement_LEFT()
            elif dirLEFT == 1:
                SNAKE.Movement_UP()
            elif dirRIGHT == 1:
                SNAKE.Movement_DOWN()
            else:
                SNAKE.Movement_RIGHT()
    
    return SNAKE.Evaluate()