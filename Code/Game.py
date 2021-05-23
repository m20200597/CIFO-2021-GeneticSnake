# -*- coding: utf-8 -*-
"""
The Snake Game used for the algorithm.
"""

class SnakeGame():
    def __init__(self, 
                 Generation = None, 
                 Child = None,
                 maximumStepsWithoutEating = 250,
                 recordGame = True, # Leave as False to save on Memory
                 BoardFillColor = (255, 255, 255), 
                 SnakeColor = (161, 221, 0), 
                 foodColor = (204, 51, 0)):
        
        import random
        import math
        import copy
        
        ### Board Parameters
        self.BoardWidth = 500
        self.BoardHeight = 500
        self.BoardBlockSize = 10
        self.BoardFillColor = BoardFillColor
        self.SnakeColor = SnakeColor
        self.FruitColor = foodColor
        self.SnakeSpeed = 15
        
        ### Game Parameters
        # Initialize Snake at the center of the board
        self.snake_X = math.floor(self.BoardWidth/2)
        self.snake_Y = math.floor(self.BoardHeight/2)
        
        # Generate random position for the fruit
        self.food_X = round(random.randrange(0, self.BoardWidth - self.BoardBlockSize) / self.BoardBlockSize) * self.BoardBlockSize
        self.food_Y = round(random.randrange(0, self.BoardWidth - self.BoardBlockSize) / self.BoardBlockSize) * self.BoardBlockSize
        
        # Initialize Snake Objects
        self.SnakeLength = 1
        self.SnakeHead = [self.snake_X,self.snake_Y]
        self.SnakeFields = [self.SnakeHead]
        
        # Game Control
        self.game_over = False
        self.game_over_explanation = ""
        self.maximumStepsWithoutEating = maximumStepsWithoutEating
        
        ### Genetic
        self.Generation = Generation
        self.Child = Child
        self.previousDistance = 9e99
        self.penalties = 0 # penalties are given for staying without eating for X amount of steps
        self.rewards = 0 # rewards are given/taken for each movement which places the snake closer/away from the fruit
        
        ### Game Results
        self.score = 0
        self.LastPredictMovement = None
        self.moves = []
        self.fruitHistory = []
        self.fruitHistory.append([copy.deepcopy(self.food_X),copy.deepcopy(self.food_Y)])
        self.ScoreHistory = []
        self.ScoreHistory.append(0)
        self.SnakeHistory = []
        self.SnakeHistory.append(copy.deepcopy(self.SnakeFields))
        self.recordGame = recordGame
        
    def Evaluate(self):
        ''' Evaluates the state of the game by returning the score, number of movements performed, 
        length of the snake, penalties and rewards.
        
        Function which is called at the end of each game to report on the performance.'''
        
        if self.game_over:
            Score = self.score
            numMovements = len(self.moves)
            lenSnake = self.SnakeLength
            penalties = self.penalties
            rewards = self.rewards
            
            return Score,numMovements,lenSnake, penalties, rewards
        else:
            print("Game not over yet.")
            return None
        
    def ComputeDistanceToFood(self):
        ''' Computes the Euclidean Distance between the Snake and fruit.
            Normalizes the distance by dividing by the corner to corner distance of the board.
        '''
        import math
        
        # Compute Distance Snake and Food
        DistanceToFood = math.sqrt((self.food_X-self.SnakeHead[0])**2 + (self.food_Y-self.SnakeHead[1])**2)
        
        maxDistance = math.sqrt((self.BoardWidth)**2 + (self.BoardWidth)**2) # distance from corner to corner
        
        NormalizedDistanceToFood = DistanceToFood / maxDistance # Normalize dist from 0 to 1
        
        return NormalizedDistanceToFood

    def ComputeAngleToFood(self):
        ''' Computes the angle in degrees from the head of the snake to the fruit.
            It corrects the angle based on the last movement performed so it is consistent with
            the direction of the movement.
            
            If LEFT, invert the angle by adding 180 degrees.
            If UP, clockwise shit by 90 degrees and counter clockwise shit by 270 degrees.
            If DOWN, clockwise shit by 270 degrees and counter clockwise shit by 90 degrees.
        '''
        import math
        
        SNAKE = self.SnakeHead
        FOOD = [self.food_X,self.food_Y]

        angle = math.atan2(FOOD[1] - SNAKE[1], FOOD[0] - SNAKE[0])

        if len(self.moves) > 0:
            if self.moves[-1] == "UP":
                corr = math.pi / 2, math.pi * 3 / 2
            elif self.moves[-1] == "LEFT":
                corr = math.pi, math.pi
            elif self.moves[-1] == "DOWN":
                corr = math.pi * 3 / 2, math.pi / 2
            else:
                corr = 0, 0 
        else:
            corr = 0, 0 

        adjusted_angle_cw = angle + corr[0]
        adjusted_angle_ccw = angle - corr[1]

        if abs(adjusted_angle_cw) < abs(adjusted_angle_ccw):
            return math.degrees(adjusted_angle_cw)
        else:
            return math.degrees(adjusted_angle_ccw)
        
    def ReturnOverview(self):
        ''' Returns an Overview of the current state of the game.
            Will serve as the input for the NN.
            
            Returns:
                float: Normalized angle to food.
                float: Normalized distance to food.
                bool : Availability of the CONTINUE space to the head of the snake.
                bool : Availability of the 90 degrees LEFT space to the head of the snake.
                bool : Availability of the 90 degrees RIGHT space to the head of the snake.
                bool : Last direction of movement was CONTINUE.
                bool : Last direction of movement was 90 degrees LEFT.
                bool : Last direction of movement was 90 degrees RIGHT.
                
            From the 2 sets of booleans, the Availability all can be 1/0, the last direction only 1 bit is 1.
        '''
        # Compute Distance Snake and Food
        NormalizedDistanceToFood = self.ComputeDistanceToFood()
        
        # Compute Upper Spot Availability
        PossibleYMove = self.snake_Y - self.BoardBlockSize # compute position of y coordinate with the possible move
        if PossibleYMove < 0: # if it violates the board
            BitUp = 0
        else:
            if [self.snake_X,PossibleYMove] in self.SnakeFields: # checks if the snake will it itself with the move
                BitUp = 0
            else:
                BitUp = 1
                
        # Compute Lower Spot Availability
        PossibleYMove = self.snake_Y + self.BoardBlockSize # compute position of y coordinate with the possible move
        if PossibleYMove > self.BoardHeight: # if it violates the board
            BitDown = 0
        else:
            if [self.snake_X,PossibleYMove] in self.SnakeFields: # checks if the snake will it itself with the move
                BitDown = 0
            else:
                BitDown = 1

        # Compute Left Spot Availability
        PossibleXMove = self.snake_X - self.BoardBlockSize # compute position of x coordinate with the possible move
        if PossibleXMove < 0: # if it violates the board
            BitLeft = 0
        else:
            if [PossibleXMove,self.snake_Y] in self.SnakeFields: # checks if the snake will it itself with the move
                BitLeft = 0
            else:
                BitLeft = 1
                
        # Compute Right Spot Availability
        PossibleXMove = self.snake_X + self.BoardBlockSize # compute position of x coordinate with the possible move
        if PossibleXMove > self.BoardWidth: # if it violates the board
            BitRight = 0
        else:
            if [PossibleXMove,self.snake_Y] in self.SnakeFields: # checks if the snake will it itself with the move
                BitRight =  0
            else:
                BitRight = 1
        
        # Gets the last movement performed by the snake and creates the booleans
        if self.LastPredictMovement is not None:
            if self.LastPredictMovement == 0:
                dirContinue = 1
                dir90left = 0
                dir90right = 0
            elif self.LastPredictMovement == 1:
                dirContinue = 0
                dir90left = 1
                dir90right = 0
            elif self.LastPredictMovement == 2:
                dirContinue = 0
                dir90left = 0
                dir90right = 1
        else:
            dirContinue = 0
            dir90left = 0
            dir90right = 0
        
        
        if len(self.moves) > 0:
            if self.moves[-1] == "LEFT":
                bitContinue = BitLeft
                bit90left = BitDown
                bit90right = BitUp
            elif self.moves[-1] == "RIGHT":
                bitContinue = BitRight
                bit90left = BitUp
                bit90right = BitDown
            elif self.moves[-1] == "UP":
                bitContinue = BitUp
                bit90left = BitLeft
                bit90right = BitRight
            elif self.moves[-1] == "DOWN":
                bitContinue = BitDown
                bit90left = BitRight
                bit90right = BitLeft
        else:
            # Snake hasn't moved yet, all available
            bitContinue = 1
            bit90left = 1
            bit90right = 1
            
        NormalizedAngleToFood = self.ComputeAngleToFood() / 180.0

        return [NormalizedAngleToFood,NormalizedDistanceToFood,bitContinue,bit90left,bit90right,dirContinue,dir90left,dir90right]
        
    def checkBoardValidity(self):
        ''' Function which detects if the current position of the Snake has violated the board limits.
            It is called after each movement to detect the game over.
        '''
        if self.snake_X >= self.BoardWidth or self.snake_X < 0 or self.snake_Y >= self.BoardHeight or self.snake_Y < 0:
            self.game_over = True
            self.game_over_explanation = "Board Limits Violated"
            return False
        else:
            return True
        
    def DidYouEatYourself(self):
        ''' Function which detects if the current position of the Snake is equal to one of its pieces.
            It is called after each movement to detect the game over.
        '''
        aux_bool = False
        for snakePart in self.SnakeFields[:-1]:
            if snakePart == self.SnakeHead:
                self.game_over = True
                self.game_over_explanation = "Snake ate itself..."
                aux_bool = True
                break
        return aux_bool

    def AreYouGettingCloser(self):
        ''' Function which computes the current distance to the food and evaluates if the snake got closer
            or further from the fruit.
            
            Will grant or remove a reward based on the output.
        '''
        currentDist = self.ComputeDistanceToFood()
        
        if currentDist < self.previousDistance:
            self.rewards = self.rewards + 1
        else:
            self.rewards = self.rewards - 1
            
        self.previousDistance = currentDist
    
    def AreYouFasting(self):
        ''' Function which evaluates if the snake is fasting for X movements.
        
            Will apply a penalty if it happens.
        '''
        import copy
        
        aux = copy.deepcopy(self.fruitHistory)
        aux.reverse()
        count = 0
        
        # logic to allow the evaluation to work if the snake has not even made X movements yet
        for i in range(1,min(len(aux),self.maximumStepsWithoutEating+1)):
            if aux[i] == aux[0]:
                count += 1
            else:
                break
        if count < self.maximumStepsWithoutEating:
            pass
        else:
            self.game_over = True
            self.penalties += 1
            self.game_over_explanation = "Snake died from not eating."
    
    def checkFruitEaten(self):
        ''' Function that checks if the last movement made the snake eat the fruit.'''
        import random
        # Compares the snake head position to the fruit position
        if self.snake_X == self.food_X and self.snake_Y == self.food_Y:
            self.score += 1
            self.SnakeLength += 1
            # Generates new fruit in a random position which is not included in the snake fields
            ValidFoodPosition = False
            while not ValidFoodPosition:
                self.food_X = round(random.randrange(0, self.BoardWidth - self.BoardBlockSize) / self.BoardBlockSize) * self.BoardBlockSize
                self.food_Y = round(random.randrange(0, self.BoardWidth - self.BoardBlockSize) / self.BoardBlockSize) * self.BoardBlockSize
                if [self.food_X,self.food_Y] not in self.SnakeFields:
                    ValidFoodPosition = True
        
    def UpdateSnake(self):
        '''Updates the snake positions'''
        import copy
        self.SnakeHead = [self.snake_X,self.snake_Y]
        self.SnakeFields.append(self.SnakeHead)
        # controls if the first position of the array needs to be deleted.
        # this happens when the snake did not eat a fruit
        if len(self.SnakeFields) > self.SnakeLength:
            del self.SnakeFields[0]
            
        if self.recordGame:
            self.SnakeHistory.append(copy.deepcopy(self.SnakeFields))
            
    def Movement_LEFT(self):
        ''' Function which moves the snake to the LEFT.'''
        import copy
        if not self.game_over:
            self.snake_X += -self.BoardBlockSize # decrease the x coordinate by the block size
            self.moves.append("LEFT")
            self.checkBoardValidity()
            self.UpdateSnake()
            self.DidYouEatYourself()
            self.AreYouGettingCloser()
            self.checkFruitEaten()
            self.AreYouFasting()
            self.fruitHistory.append([copy.deepcopy(self.food_X),copy.deepcopy(self.food_Y)])
            if self.recordGame:
                self.ScoreHistory.append(copy.deepcopy(self.score))
        else:
            print(f"Your game has ended. Final Score {self.score} , Number of Moves {len(self.moves)}")
    def Movement_RIGHT(self):
        ''' Function which moves the snake to the RIGHT.'''
        import copy
        if not self.game_over:
            self.snake_X += self.BoardBlockSize # increase the x coordinate by the block size
            self.moves.append("RIGHT")
            self.checkBoardValidity()
            self.UpdateSnake()
            self.DidYouEatYourself()
            self.AreYouGettingCloser()
            self.checkFruitEaten()
            self.AreYouFasting()
            self.fruitHistory.append([copy.deepcopy(self.food_X),copy.deepcopy(self.food_Y)])
            if self.recordGame:
                self.ScoreHistory.append(copy.deepcopy(self.score))
        else:
            print(f"Your game has ended. Final Score {self.score} , Number of Moves {len(self.moves)}")
    def Movement_DOWN(self):
        ''' Function which moves the snake DOWN.'''
        import copy
        if not self.game_over:
            self.snake_Y += self.BoardBlockSize # increase the y coordinate by the block size
            self.moves.append("DOWN")
            self.checkBoardValidity()
            self.UpdateSnake()
            self.DidYouEatYourself()
            self.AreYouGettingCloser()
            self.checkFruitEaten()
            self.AreYouFasting()
            self.fruitHistory.append([copy.deepcopy(self.food_X),copy.deepcopy(self.food_Y)])
            if self.recordGame:
                self.ScoreHistory.append(copy.deepcopy(self.score))
        else:
            print(f"Your game has ended. Final Score {self.score} , Number of Moves {len(self.moves)}")
    def Movement_UP(self):
        ''' Function which moves the snake UP.'''
        import copy
        if not self.game_over:
            self.snake_Y += -self.BoardBlockSize # decrease the y coordinate by the block size
            self.moves.append("UP")
            self.checkBoardValidity()
            self.UpdateSnake()
            self.DidYouEatYourself()
            self.AreYouGettingCloser()
            self.checkFruitEaten()
            self.AreYouFasting()
            self.fruitHistory.append([copy.deepcopy(self.food_X),copy.deepcopy(self.food_Y)])
            if self.recordGame:
                self.ScoreHistory.append(copy.deepcopy(self.score))
        else:
            print(f"Your game has ended. Final Score {self.score} , Number of Moves {len(self.moves)}")
            
    def TimeWalker(self,FOLDER = "", BASENAME = "", record = True):
        ''' Function that creates a pygame display and plays the game as it happened.'''
        import pygame
        from PIL import Image
        import time
        import numpy as np
        import cv2
        
        if record and FOLDER == "":
            return "No folder specified."
        
        GameSpeed = 0.05
        messageColor = (0,0,0)
        
        pygame.init()
        
        display = pygame.display.set_mode((self.BoardWidth, self.BoardHeight))
        pygame.display.set_caption(f'Snake - Generation {self.Generation} - Child {self.Child}')
        font_style = pygame.font.SysFont("bahnschrift", 15)
        frames = []
        
        for i,move in enumerate(self.SnakeHistory):
            display.fill(self.BoardFillColor)
            mesg = font_style.render(f"Position {i+1}", True, messageColor)
            display.blit(mesg, [self.BoardWidth * 0.8, self.BoardHeight * 0.005])
            mesg = font_style.render(f"Score {self.ScoreHistory[i]}", True, messageColor)
            display.blit(mesg, [self.BoardWidth * 0.8, self.BoardHeight * 0.03])
            for x in self.SnakeHistory[i]:
                pygame.draw.rect(display, self.SnakeColor, [x[0], x[1], self.BoardBlockSize, self.BoardBlockSize])
            pygame.draw.rect(display, self.FruitColor, [self.fruitHistory[i][0], self.fruitHistory[i][1], self.BoardBlockSize, self.BoardBlockSize])
            pygame.display.update()
            
            if record:
                data = pygame.image.tostring(display, 'RGBA')
                img = Image.frombytes('RGBA', (self.BoardWidth, self.BoardHeight), data)
                img = img.convert('RGB')
                frames.append(np.array(img))
                
            time.sleep(GameSpeed)
        pygame.display.quit()
        if record:
            size = (frames[0].shape[1], frames[0].shape[0])
            out = cv2.VideoWriter(f'{FOLDER}\\{BASENAME}-Gen{self.Generation}-{self.Child}.mp4', 
                                  cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
            for i in range(len(frames)):
                out.write(frames[i])
            out.release()