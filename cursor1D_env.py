import random
import numpy as np
import random
import pygame
import sys

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


class cursor1D_env():


    #Constructor parameters are for noise values on the reward
    #State space is continuous from 0 to 1 position of moving cursor which is randomly initialized
    #Action space is move cursor left = 1, right = 2, or stop = 0
    #Reward uses true positive rate bernouli if moving towards the goal or within 0.025 distance of goal, 
    #otherwise gives a rewqard using false positive rate
    #Goal is located at random position from 0.375 to 0.625
    def __init__(self, truePositiveRate = 0.7, falsePositiveRate = 0.3):
        
        self.truePositiveRate = truePositiveRate
        self.falsePositiveRate = falsePositiveRate
        
        self.goal_location = random.uniform(0.375, 0.625)
        self.state = random.random()
        
        self.screen = None
        self.r = 0
        self.a = 0 
        self.d = False
        
        self.hit = 0
        self.miss = 0
       

    def reset(self):
        self.state = random.random()
        self.r = 0
        self.a = 0 
        self.d = False
        
        return self.state
    
    def reset_left(self):
        self.state = 0
        self.r = 0
        self.a = 0 
        self.d = False
        
        return self.state
    
    def reset_right(self):
        self.state = 1
        self.r = 0
        self.a = 0 
        self.d = False
        
        return self.state

    def render(self, size = [800,400], display_sr = False, display_hm = False):
        
        #Create screen if not there
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(size)
            self.size = size
            pygame.display.set_caption("Cursor 1D Environment")

        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                pygame.display.quit()
                pygame.quit()
                self.screen = None
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if abs(self.state - self.goal_location) < 0.025:
                        self.hit += 1
                    else:
                        self.miss += 1
                    
        #Render the scene:
        self.screen.fill(WHITE)
        
        #Render the line
        y_pos = int(0.5*self.size[1])
        x_left_line = int(0.1*self.size[0])
        lengthOfLine = int(self.size[0]*0.8)
        pygame.draw.line(self.screen, BLUE, [x_left_line, y_pos], [x_left_line + lengthOfLine, y_pos], 5)
        
        #Render the goal
        x_goal_pos = int(x_left_line + int(self.goal_location*lengthOfLine))
        pygame.draw.circle(self.screen, RED  , [ x_goal_pos,y_pos], int(0.025*lengthOfLine))
        pygame.draw.circle(self.screen, WHITE, [ x_goal_pos,y_pos], int(0.020*lengthOfLine))
        pygame.draw.circle(self.screen, RED  , [ x_goal_pos,y_pos], int(0.015*lengthOfLine))
        pygame.draw.circle(self.screen, WHITE, [ x_goal_pos,y_pos], int(0.010*lengthOfLine))
        pygame.draw.circle(self.screen, RED  , [ x_goal_pos,y_pos], int(0.005*lengthOfLine))


        #Render the cursor
        x_cursor_left  = int(x_left_line + lengthOfLine*self.state - 0.015*lengthOfLine)
        x_cursor_right = int(x_left_line + lengthOfLine*self.state + 0.015*lengthOfLine)
        y_cursor_up    = int(y_pos + 0.015*lengthOfLine)
        y_cursor_down  = int(y_pos - 0.015*lengthOfLine)
        
        pygame.draw.line(self.screen, BLACK, [x_cursor_left, y_cursor_up], [x_cursor_right, y_cursor_down], 7)
        pygame.draw.line(self.screen, BLACK, [x_cursor_left, y_cursor_down], [x_cursor_right, y_cursor_up], 7)
        
        font = pygame.font.SysFont('Calibri', 25, False, False)
        
        #Display state and reward info in the upper left corner
        if display_sr:
            text = font.render("S = {}, A = {}, R = {}, D = {}".format(self.state, self.a, self.r, self.d), True, BLACK)
            self.screen.blit(text, [20, 20])
            
        #Display state and reward info in the upper righ corner
        if display_hm:
            text = font.render("{}/{}".format(self.hit, self.miss), True, BLACK)
            self.screen.blit(text, [self.size[0]-100, 20])
        
        pygame.display.flip()
    
    #Gets a random action with 1/3 prob for all
    def random_action(self):
        return random.randint(0, 2)
    
    #This should be overwritten to change the generalized reward!!
    #Returns the generalized/heuristic reward
    #The default here is give reward if state is between 0.375 to 0.625
    #Or if moving towards that region
    
    #s is state after taking action a from last_s
    def getHeuristicReward(self, s,  a, last_s):
        
        #Between 0.625 and 0.375
        if s <= 0.625 and s >= 0.375:
            r_huer = 0.25
        
        
        else:
            #Moving towards the goal
            if (s < 0.375 and a == 2) or (s > 0.625 and a == 1):
                r_huer = 0.25
            else:
                r_huer = 0
                
        return r_huer;
    
    #Can be overwritten to define a new done condition.
    #Default condition is current state is within distance of 0.025 of the goal and action is 0
    def doneCondition(self, s, a, last_s):
        if abs(self.state - self.goal_location) < 0.025 and a == 0:
            return True
        else:
            return False
    
    def step(self, action, l):
        
        if l > 1 or l < 0 :
            raise Exception("l must be between 0 and 1.")
        
        last_s = self.state
        
        #move the current state left or right depending on the action applied
        if action == 1:
            self.state = self.state - 0.001
        elif action == 2:
            self.state = self.state + 0.001
          
        #Clamp the state to make sure we do not go out of bounds
        if self.state > 1:
            self.state = 1
        elif self.state < 0:
            self.state = 0
            
        #Get the real noisy reward
        sample = random.random()
            
        #Close to the goal
        if abs(self.state - self.goal_location) < 0.025:
            if sample < self.truePositiveRate:
                r_real = 1
            else:
                r_real = 0
        #Moving towards the goal
        elif (self.state - self.goal_location > 0 and action == 1) or (self.state - self.goal_location < 0 and action == 2):
            if sample < self.truePositiveRate:
                r_real = 1
            else:
                r_real = 0
        else:
            if sample < self.falsePositiveRate:
                r_real = 1
            else:
                r_real = 0
            
        #Get our heuristicly defined reward.
        r_huer = self.getHeuristicReward(self.state,  action, last_s)
        
        r = l*r_real + (1-l)*r_huer
        
        self.r = r
        self.a = action
            
        d = self.doneCondition(self.state, action, last_s)
        self.d = d
        
        return (self.state, r, d, {})