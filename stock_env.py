#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:36:30 2022

@author: Stephen
"""

"""
Do not change your TabularQLearner.py code.  It contains generic Q-Learning that does not care what problem it solves.
Use exactly the same indicators as in the prior technical strategy project to produce comparable results.
In sample testing should be performed by calling your test() method with the same dates used to train.
Out of sample testing should be performed by calling your test() method with dates in the future of those used to train.
Only perform out of sample testing at the very end, once you are satisfied with your in sample results.

State should be (day, shares, indicator_1, indicator_2, indicator_3)
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  
from TabularQLearner import TabularQLearner
import tech_ind 
import sys
import timeit
import datetime


class StockEnvironment:

  def __init__ (self, fixed = None, floating = None, starting_cash = None, share_limit = None):
    self.shares = share_limit
    self.fixed_cost = fixed
    self.floating_cost = floating
    self.starting_cash = starting_cash

  def prepare_world (self, start_date, end_date, symbol):
    """
    Read the relevant price data and calculate some indicators.
    Return a DataFrame containing everything you need.
    """
    price_data = tech_ind.get_data(start_date, end_date, symbols = [symbol], include_spy=False)
    volume_data = tech_ind.get_data(start_date, end_date, symbols = [symbol], column_name = 'Volume', include_spy=False)
    williams = tech_ind.Williams_Percentage_Range(price_data)
    bb = tech_ind.Bollinger_Bands(price_data)
    obv = tech_ind.On_Balance_Volume(price_data, volume_data)
    world = price_data.copy()
    world.columnes = 'Price'
    world['Williams Percent Range'] = williams
    world['Bollinger Band Percentage'] = (world['Price'] - bb['SMA'])/(bb['Top Band' - bb['Bottom Band']])
    world['OBV Normalized'] = obv/(abs(obv).rolling(window=5, min_periods=5).sum()) * 100 # scaled between -100 and 100
    return world
    

  def calc_state (self, df, day, holdings):
    """ Quantizes the state to a single number. """
    
    world = df
    current_williams = df[day]['Williams Percent Range'] # 0 to -100
    current_bbp = df[day]['Bollinger Band Percentage']
    current_obv = df[day]['OBV Normalized']
    state = ''
    state += str((abs(current_williams)-10) % 10)   
    state += str(((current_obv + 100) % 20))
    if current_bbp < 10:
        state += '0'
    elif current_bbp >= 90:
        state += '9'
    else:
        state += str(current_bbp % 10)
    
    if holdings < 0: 
        state += '0'
    if holdings == 0:
        state += '1'
    if holdings > 1:
        state += '2'

    return state

  def train_learner( self, start = None, end = None, symbol = None, trips = 0, dyna = 0,
                     eps = 0.0, eps_decay = 0.0 ):
    """
    Construct a Q-Learning trader and train it through many iterations of a stock
    world.  Store the trained learner in an instance variable for testing.

    Print a summary result of what happened at the end of each trip.
    Feel free to include portfolio stats or other information, but AT LEAST:

    Trip 499 net result: $13600.00
    """
    
    world_size = 10*10*10*3
    
    if dyna > 0:
      learner = TabularQLearner(states=world_size, actions = 3, epsilon=eps,epsilon_decay=eps_decay, dyna=dyna)
    else:
      learner = TabularQLearner(states=world_size, actions = 3, epsilon=eps,epsilon_decay=eps_decay)
    
    
    
    reward = #daily portfolio change 
    
    pass


  def test_learner( self, start = None, end = None, symbol = None):
    """
    Evaluate a trained Q-Learner on a particular stock trading task.
    Print a summary result of what happened during the test.

    Feel free to include portfolio stats or other information, but AT LEAST:

    Test trip, net result: $31710.00
    Benchmark result: $6690.0000
    """
    pass