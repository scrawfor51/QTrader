#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:13:32 2022

@author: Stephen
Financial Machhine Learning 
"""

import numpy as np
import random 

class TabularQLearner:

    
    # need to incorporate epsilon greedy
    # we want to try learn policy without explicitly modeling the world (T, R)
    # Q(s, a) --> R maps every state-action pair to a real number that is the expected sum of current and future rewards from taking action a from state s
    # initalize q table randomly so always have some estimate of the q value for everry state-action pair 
    # q value must be summ of immediate reward for engaging a stateion-actiono pair and the discounted future rewards we expect to receive after taking that stateaction paiir 
    # policy(s) = argmaxQ[s, a] for all a from the state s; run until convergence i.e. q values do not change 
    
    
    """
    Construct class instance
    
    @param states: The number of distinct states 
    @param actions: The number of distinct actions 
    @param alpha: Learning rate
    @param gamme: Discount rate
    @param epsilon: Random action rate
    @param epsilon_decay: The rate at which the random action rate decreases after each random action
    """
    def __init__ (self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9, epsilon = 0.98, epsilon_decay = 0.999, dyna=0):
        # Store all the parameters as attributes (instance variables).
        # Initialize any data structures you need.
        self.experience_history = []
        self.q_table = np.random.rand(states, actions) * 0.0000000001 # initalize a table of q-values each starting as basically 0
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.dyna = dyna
        self.last_move = []
    
    
    
    def train (self, s, r):
        # Receive new state s and new reward r.  Update Q-table and return selected action.
        # Consider: The Q-update requires a complete <s, a, s', r> tuple.
        #           How will you know the previous state and action?
            
        # grab the last state and action 
        
        last_experience = self.last_move
        
        old_s = last_experience[0]
        old_a = last_experience[1]
        
        # q-update 
        weighted_current_estimate = (1 - self.alpha)*(self.q_table[old_s, old_a]) # 1-alpha * the old q value for the state and action we took
        weighted_new_estimate = self.alpha*(r + self.gamma*(max(self.q_table[s, :]))) # alpha * (the reward we got for taking the state and action + the discounted best results from the new state)
        self.q_table[old_s, old_a] = weighted_current_estimate + weighted_new_estimate
        
        # pick new action:
        epsilon_checker = random.random()
       
        # find the new action we plan to take from the new state
        if epsilon_checker < self.epsilon:
        
            a = random.randrange(0, self.q_table.shape[1])
            self.epsilon = self.epsilon * self.epsilon_decay
            
        else:
            a = self.q_table[s,:].argmax(axis=0) # best action in the current state
        

        self.experience_history.append([old_s, old_a, s, r])
        experience_index = np.random.randint(0, len(self.experience_history), (self.dyna,))
        
        for i in range(self.dyna): 
            
            exp = experience_index[i]
            selected_experience = self.experience_history[exp]

            dyna_s = selected_experience[0]
            dyna_a = selected_experience[1]
            dyna_s_prime = selected_experience[2]
            dyna_r = selected_experience[3]
            weighted_current_estimate = (1 - self.alpha)*(self.q_table[dyna_s, dyna_a]) # 1-alpha * the old q value for the state and action we took
            weighted_new_estimate = self.alpha*(dyna_r + self.gamma*(max(self.q_table[dyna_s_prime, :]))) # alpha * (the reward we got for taking the state and action + the discounted best results from the new state)
            self.q_table[dyna_s, dyna_a] = weighted_current_estimate + weighted_new_estimate
        
        self.last_move = [s, a]
         
        return a
    
    
    def test (self, s):
        # Receive new state s.  Do NOT update Q-table, but still return selected action.
        #
        # This method is called for TWO reasons: (1) to use the policy after learning is finished, and
        # (2) when there is no previous state or action (and hence no Q-update to perform).
        #
        # When testing, you probably do not want to take random actions... (What good would it do?)
        
        a = self.q_table[s, :].argmax(axis=0) # best action in the current state
        self.last_move = [s, a]
        
        return a