from gym.envs.toy_text import FrozenLakeEnv
import random

class gridWorld_env(FrozenLakeEnv):
    
    #There are no holes in this environment (nor nothing to handle that case...)
    #Also it is not slippery. This is not investigating transition models, rather noisy rewards.
    
    #I did not add a goal block because I do not wnat the environment to stop when the goal is reached
    #We do not know the goal nor when the environment is completed
    def __init__(self, truePositiveRate = 0.7, falsePositiveRate = 0.3):
        desc = ["SFFF",
                "FFFF",
                "FFFF",
                "FFFF"]
        
        super(gridWorld_env,self).__init__(desc=desc, is_slippery=False)
        
        self.truePositiveRate = truePositiveRate
        self.falsePositiveRate = falsePositiveRate
    
    #This should be overwritten to change the generalized reward!!
    #Returns the generalized/heuristic reward
    #The default here gives a hueristic reward of 1 for bottom right quadrant
    
    #s is state after taking action a from last_s
    def getHeuristicReward(self, s,  a, last_s):
        #We know the goal is somewhere the bottom right quadrant
        
        #If in bottom right quadrant get reward of 1
        if s == 15 or s == 14 or s == 11 or s == 10:
            r_huer = 1
        
        #Otherwise only give reward of 1 if moving to bottom right quadrant
        else:
            if (a == 1 or a == 2) and last_s != s:
                r_huer = 1
            else:
                r_huer = 0
                
        return r_huer;
         
    #We modify the reward output from the step function
    #l input should be a number from 0 to 1. It defines 
    def step(self, a, l):
        last_s = self.s
        
        if l > 1 or l < 0 :
            raise Exception("l must be between 0 and 1.")
        
        s, _, d, info = super(gridWorld_env, self).step(a)            
        
        #Get the real noisy reward
        sample = random.random()
        
        #If we reached the goal give a reward of 1
        if s == 15:
            if sample < self.truePositiveRate:
                r_real = 1
            else:
                r_real = 0

        #If going to the goal, the true reward should be 1
        elif (a == 1 or a == 2) and last_s != s:
            if sample < self.truePositiveRate:
                r_real = 1
            else:
                r_real = 0
                
        #Means we are going away from the goal.
        #Or stuck on a wall
        else:
            if sample < self.falsePositiveRate:
                r_real = 1
            else:
                r_real = 0
        
        #Get our heuristicly defined reward.
        r_huer = self.getHeuristicReward(s,  a, last_s)
        
        r = l*r_real + (1-l)*r_huer
        
        if s == 15:
            d = True
        else:
            d = False
        
        return (s, r, d, info)