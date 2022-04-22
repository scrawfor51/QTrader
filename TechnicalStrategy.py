    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 22:42:13 2022
@author: Stephen
"""

"""
Created on Thu Apr 15
"""

import tech_ind 
import pandas as pd
import numpy as np
import backtester_manual_trading as backtester
from matplotlib import cm
import matplotlib.pyplot as plt
from OracleStrategy import OracleStrategy

class TechnicalStrategy:
    def __init__(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def train(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def test(self, start_date='2009-01-01', end_date='2010-12-31', symbol = 'DIS', starting_cash = 200000):
        
        lookback = 14
        
        price = tech_ind.get_data(start_date, end_date, symbols=[symbol], include_spy=(False))
        ### Use the three indicators to make some kind of trading decision for each day.
        trades = price.copy()
        trades.loc[:,:] = np.NaN
        volume = tech_ind.get_data(start_date, end_date, symbols=[symbol], column_name = 'Volume', include_spy=(False))
        obv = tech_ind.On_Balance_Volume(price, volume)
        williams_per  = tech_ind.Williams_Percentage_Range(price, lookback)
        bb = tech_ind.Bollinger_Bands_Percentage(price, lookback)     
        
        obv_delta = obv.copy()
        obv_delta = abs(obv - obv.shift(-1))
        obv_delta.iloc[0] = 0
        obv_delta = obv_delta.ffill()
        
        obv_avg = obv_delta.rolling(window=lookback,min_periods=lookback).mean()
        obv_avg = obv_avg.bfill()
        obv_avg = obv_avg.ffill()
        
        obv_std = obv.rolling(window=lookback,min_periods=lookback).std()
       
        # Create a binary (0-1) array showing when price is above SMA-14.
        sma_cross = pd.DataFrame(0, index=bb.index, columns= ["SMA"])
        
        bb_ratios = sma_cross.copy()
        
        bb_ratios["Price"] = bb["Price"] / bb["Price"]
        bb_ratios["SMA"] = bb["Price"] / bb["SMA"]        
        
        sma_cross[bb_ratios["SMA"] >= 1] = 1
        
        
        #Create a binary (0-1) array showing when OBV is within 10% of OBV-lookback
        obv_trade = obv_delta.copy()
        obv_trade.columns=["DIS"]
        
        obv_trade[obv_trade > obv_avg] = 1
        obv_trade[obv_trade != 1 ] = 0
        obv_trade.columns=["OBV Trade"]
        
        # Turn that array into one that only shows the crossings (-1 == cross down, +1 == cross up).
        sma_cross.iloc[1:] = sma_cross.diff().iloc[1:]
        sma_cross.iloc[0] = 0         
           
        #Breakout
        # If Price over Top and Williams Not Over Bought and OBV Good
        trades[(bb["Top Band"] < bb["Price"]) & ((williams_per["Williams Percentage"] < -20) & (obv_trade["OBV Trade"] == 1)) ] = 1000
        # If Price under Bottom and Williams Not Over Sold and OBV Good
        trades[(bb["Bottom Band"] > bb["Price"]) & ((williams_per["Williams Percentage"] > -80) & (obv_trade["OBV Trade"] == 1))] = -1000     
        
        #Return-to-Mean within the Bollinger Bands
        # Price over SMA, below Top Band and above 80% Bollinger Band Percentage -- return to mean  
        trades[(bb["Top Band"] < bb["Price"]) & ((williams_per["Williams Percentage"] > -20) | (obv_trade["OBV Trade"] == 0))] = -1000
        # Price below top band and over sold
        trades[(bb["Bottom Band"] > bb["Price"]) & ((williams_per["Williams Percentage"] < -80) | (obv_trade["OBV Trade"] == 0))] = 1000
        
        # Exit when we revert back to the mean
        trades[(sma_cross != 0)] = 0               
        
        #Close trade to avoid skewing the calculations by shorting on last day
        trades.iloc[-1] = 0
        
        # Forward fill NaNs with previous values, then fill remaining NaNs with 0.
        trades.ffill(inplace=True)
        trades.fillna(0, inplace=True)
        
        trades.iloc[1:] = trades.diff().iloc[1:]
        trades.iloc[0] = 0        
        
        # And more importantly, drop all rows with no non-zero values (i.e. no trades).
        trades = trades.loc[(trades != 0).any(axis=1)]
        
        return trades

if __name__ == "__main__":    
    start ='2008-01-01'
    end = '2009-12-31'
    sym = "DIS"
    data = tech_ind.get_data(start_date = start, end_date = end, symbols =[sym],  include_spy=False)
   
    OS = OracleStrategy()
    oracle = OS.test(start_date = start, end_date = end)
    oracle_portfolio = backtester.assess_strategy_dataframe(oracle, start, end, starting_value = 200000, fixed_cost = 0, floating_cost = 0)
    print("ORACLE STATS: ")
    backtester.calc_portfolio(oracle_portfolio)
    
    baseline = data.copy()
    baseline.loc[:,:] = np.NaN
    baseline.iloc[0,0] = 1000
    baseline_portfolio = backtester.assess_strategy_dataframe(baseline, start, end, starting_value = 200000, fixed_cost = 0, floating_cost = 0)
    print("BASELINE STATS: ")
    backtester.calc_portfolio(baseline_portfolio)
    
    TS = TechnicalStrategy()
    tech_trader = TS.test(start_date = start, end_date = end, symbol = sym)
    tech_portfolio = backtester.assess_strategy_dataframe(tech_trader, start, end, starting_value = 200000)
    print("TECHNICAL STRATEGY STATS: ")
    backtester.calc_portfolio(tech_portfolio)
    
    base_returns = baseline_portfolio.copy() - 200000
    base_returns.columns = ['Baseline']
    base_returns['Tech Portfolio'] =  tech_portfolio - 200000
    plot = base_returns.plot(title='Baseline vs. Tech Strategy', colormap=cm.Accent)
    plot.grid()
    
    
    
    #Print the short and long lines
    tech_trader['position'] = tech_trader[sym].cumsum()
    for day in tech_trader.index:
        #blue when strategy is long
        if tech_trader.loc[day,'position'] > 0:
            plt.axvline(x=day, color = 'blue') 
        #red when strategy is short
        elif tech_trader.loc[day,'position'] < 0:
            plt.axvline(x=day, color = 'red') 
        #black when closing a position
        else:
            plt.axvline(x=day, color = 'black')
    
    plt.show()    