#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 21:24:11 2022

@author: Stephen

Note: Read many tutorials for matplotlib to get plots how we wanted. We take no credit for the developent of the plotting code. 
We just copied the necessary parts from tutorials and the docs. 
"""


"""

Functions checked: get_data; sma; sma_ratio; momentum; Bollinger Bands, BBP, RSI, Aroon, stochastic oscillator 

Still unsure: EMA, MACD (Requires EMA), Signal (Requires EMA), OBV (Needs volume data)
"""

import tech_ind
import os
import pandas as pd 
from matplotlib import pyplot as plt 
import math
import numpy as np
from backtest import get_data as gd
from matplotlib import pyplot as plt
from matplotlib import cm

def plot_obv(start_date='2018-01-01', end_date = '2019-12-31', symbols=['DIS'], include_spy=(False)):
    
    processed_data_price = tech_ind.get_data(start_date='2014-01-01', end_date='2014-12-01', symbols=['DIS'], include_spy=(False))
    processed_data_vol = tech_ind.get_data(start_date='2014-01-01', end_date='2014-12-01', symbols=['DIS'], column_name = 'Volume', include_spy=(False))
    obv = tech_ind.On_Balance_Volume(processed_data_price, processed_data_vol)
    
    obv.columns = ["On-Balance Volume"]
    
    obv["Daily Volume"] = processed_data_vol
    obv["Price"] = processed_data_price
    norm_obv = processed_data_price.copy()
    norm_obv.columns = ["Price"]
    norm_obv["Scaled OBV"]  = obv["On-Balance Volume"] / 10000000 + 100
    plot = norm_obv.plot(figsize=(15, 12), title='On-Balance Volume vs. Price', colormap=cm.Dark2)
    plot.set_ylabel('Price per Share')   
    plot.grid()
    
    # connect peaks 
    norm_obv["OBV Peaks"] = norm_obv["Scaled OBV"][(norm_obv["Scaled OBV"].shift(1) > norm_obv["Scaled OBV"]) & (norm_obv["Scaled OBV"].shift(-1) < norm_obv["Scaled OBV"])]
    norm_obv["Price Peaks"] = norm_obv["Price"][(norm_obv["Price"].shift(1) > norm_obv["Price"]) & (norm_obv["Price"].shift(-1) < norm_obv["Price"])]
    plot.plot(norm_obv["Price Peaks"].dropna(), linestyle='-', marker='.', alpha=0.5)
    plot.plot(norm_obv["OBV Peaks"].dropna(), linestyle='-', marker='.', alpha=0.5)


def plot_williams(start_date='2018-01-01', end_date = '2019-12-31', symbols=['DIS'], include_spy=(False)):
    processed_data_price = tech_ind.get_data(start_date, end_date, symbols, include_spy=False)
    williams_per  = tech_ind.Williams_Percentage_Range(processed_data_price)
    williams_per.columns = ["Price", "Williams Percent"]
    williams_per['Price'] = processed_data_price.values
    williams_per["Overbought"] = -20 # overbought
    williams_per["Oversold"] = -80 # oversold
 
   
    # split code
    plot = williams_per.plot(figsize=(15, 12), title='Price vs. Williams Percent Range', colormap=cm.Dark2)
    plot.set_ylabel("Price Per Share")
    plot2 = plot.twinx()
    plot2.set_ylim(plot.get_ylim())
    plot2.set_yticks(range(-100, 10, 10))
    plot2.set_ylabel("Percent Range")
    plot2.yaxis.set_label_coords(1.1, .25)
    
    
    #overlay code 
    #plot = williams_per.plot(figsize=(15, 12), title='Price vs. Williams Percent Range', secondary_y=["Overbought", "Oversold", "Williams Percent"], colormap=cm.Dark2)
    #plot.set_ylabel("Price Per Share")
    #plot.right_ax.set_ylabel("Percent Range")
    
    
    plot.grid()
    

# Create a binary (0-1) array showing when price is above SMA-14.

    

def plot_bb(start_date='2018-01-01', end_date = '2019-12-31', symbols=['DIS'], include_spy=(False)):
    
    processed_data_price = tech_ind.get_data(start_date='2019-01-01', end_date='2019-12-01', symbols=['DIS'], include_spy=(False))
    bb = tech_ind.Bollinger_Bands(processed_data_price, 14)
    
    sma_crossovers = pd.DataFrame(0, index=bb.index, columns=["Price"])
    bb_bottom_crossovers = sma_crossovers.copy()
    bb_top_crossovers = sma_crossovers.copy()

    bb_ratios = sma_crossovers.copy()

    bb_ratios["Price"] = bb["Price"] / bb["Price"]
    bb_ratios["SMA"] = bb["Price"] / bb["SMA"]
    bb_ratios["Top Band"] = bb["Price"] / bb["Top Band"]
    bb_ratios["Bottom Band"] = bb["Price"] / bb["Bottom Band"]
    
    sma_crossovers[bb_ratios["SMA"] >= 1] = 1
    
    bb_bottom_crossovers[bb_ratios["Bottom Band"] >= 1] = 1
    
    bb_top_crossovers[bb_ratios["Top Band"] >= 1] = 1
    
    # Turn that array into one that only shows the crossings (-1 == cross down, +1 == cross up).
    sma_crossovers.iloc[1:] = sma_crossovers.diff().iloc[1:]
    sma_crossovers.iloc[0] = 0
    
    bb_bottom_crossovers.iloc[1:] = bb_bottom_crossovers.diff().iloc[1:]
    bb_bottom_crossovers.iloc[0] = 0
    
    bb_top_crossovers.iloc[1:] = bb_top_crossovers.diff().iloc[1:]
    bb_top_crossovers.iloc[0] = 0
    
    sma_intersection_up_rows = bb_ratios.index[sma_crossovers["Price"] == 1]
    sma_intersection_down_rows = bb_ratios.index[sma_crossovers["Price"] == -1]
    
    bb_top_intersection_up_rows = bb_ratios.index[bb_top_crossovers["Price"] == 1]
    bb_top_intersection_down_rows = bb_ratios.index[bb_top_crossovers["Price"] == -1]
    
    bb_bottom_intersection_up_rows = bb_ratios.index[bb_bottom_crossovers["Price"] == 1]
    bb_bottom_intersection_down_rows = bb_ratios.index[bb_bottom_crossovers["Price"] == -1]
    
    
    plot = bb.plot(figsize=(15, 12), title='Bollinger Bands')
    plot.set_ylabel('Price Per Share')
    plot.grid()

   
    # plot.plot(sma_intersection_up_rows, bb["Price"][sma_intersection_up_rows], 'rD')
    # plot.plot(sma_intersection_down_rows, bb["Price"][sma_intersection_down_rows], 'bD')
    
    # plot.plot(bb_top_intersection_up_rows, bb["Price"][bb_top_intersection_up_rows], 'r.')
    # plot.plot(bb_top_intersection_down_rows, bb["Price"][bb_top_intersection_down_rows], 'b.')
    
    # plot.plot(bb_bottom_intersection_up_rows, bb["Price"][bb_bottom_intersection_up_rows], 'r*')
    # plot.plot(bb_bottom_intersection_down_rows, bb["Price"][bb_bottom_intersection_down_rows], 'b*')
    
    
    
if __name__ == "__main__":
    plot_obv()
    
