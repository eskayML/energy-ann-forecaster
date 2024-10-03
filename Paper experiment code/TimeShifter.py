# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:25:43 2020

@author: Erik Heilmann
"""
def TimeShift(Input, Target):
        """Time Shifting by the lag of 4"""
        #create lag series
        Target_temp = Target.copy(deep=True)
        shift1=Target_temp.shift(periods=1)
        shift2=Target_temp.shift(periods=2)
        shift3=Target_temp.shift(periods=3)
        shift4=Target_temp.shift(periods=4)
        
        #copy input
        Input_temp=Input.copy(deep=True)
        #complement with time lags
        Input_temp['lag1']=shift1
        Input_temp['lag2']=shift2
        Input_temp['lag3']=shift3
        Input_temp['lag4']=shift4
        #only use complete data points
        Input_temp=Input_temp[4:]
        
        #only use targets with complete inputs
        Target_temp=Target_temp[4:]
           
        return Input_temp, Target_temp

