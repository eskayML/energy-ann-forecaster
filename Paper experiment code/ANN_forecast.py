# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:32:36 2020

@author: Erik Heilmann
"""
import numpy as np
import pandas as pd


def ann_prediction (model,forecast_horizon, train_input, train_target, 
                    test_input, test_target, train_true, test_true, scaler):
    '''Function for static and dynamic prediction and scoring of selected ANN'''
    '''Input: model, forecast horizin, scaled input and target for train and testing, Unscaled target of train and test set'''
    '''Output: static and dynamic score of train set, static and dynamic score of test set'''
    '''Note: there must be a (re-)scaler of target data'''
    #fit model on training data
    np.random.seed(123)
    model.fit(train_input, train_target)
    
    ##static prediciton
    
    #for training set
    pred_train_scaled=model.predict(train_input)
    #re-scale
    pred_train=scaler.inverse_transform(pred_train_scaled)
    #set time stamp and columnname
    # pred_train=pd.DataFrame(pred_train).set_index(train_true.index)
    
    #test set
    pred_test_scaled=model.predict(test_input)
    #re-scale
    pred_test=scaler.inverse_transform(pred_test_scaled)
    #set time stamp and columnname
    # pred_test=pd.DataFrame(pred_test).set_index(test_true.index)
    
    
    ##dynamic prediction
    
    
    #for train set:
    
    #empty list for dynamic forecast values
    forecast_dyn=[]
    #make predictions over forecast horizon, then step over one forecast horizon
    for start in np.arange(0,len(train_target)-forecast_horizon,forecast_horizon):
        #only take relevant input data   
        input_temp_horizon=train_input[start: start+forecast_horizon].copy(deep=True)
        #empty list for forecast over one horizon    
        forecast_over_horizon=[]
        for step in range(0,forecast_horizon):
            #for first step, take the true data
            if step==0:
                input_temp = input_temp_horizon.iloc[step:step+1].copy()
            # for step 2 until end of forecast horizon: use predicted values 
            # of load with time lag 1-4
            else:
                input_temp_old=input_temp.copy()
                input_temp = input_temp_horizon.iloc[step:step+1].copy()
                input_temp.reset_index(drop=True,inplace=True)
                input_temp.loc[:,'lag1']=forecast_over_horizon[step-1]
                input_temp.loc[:,'lag2']=input_temp_old.loc[:,'lag1'].values
                input_temp.loc[:,'lag3']=input_temp_old.loc[:,'lag2'].values
                input_temp.loc[:,'lag4']=input_temp_old.loc[:,'lag3'].values
            #predict with relevant input data (only one time-step prediction per iteration)
            pred_step=model.predict(input_temp)
            forecast_over_horizon.append(pred_step)
    #append forecast over horizon to dynamic forecast
        forecast_over_horizon=np.ravel(forecast_over_horizon)
        forecast_dyn.append(forecast_over_horizon)
    #transform into one forecast float
    forecast_dyn=np.ravel(forecast_dyn)
    
    #re-scale
    pred_train_dyn=scaler.inverse_transform(forecast_dyn)
    #set time stamp and columnname
    #pred_train_dyn=pd.DataFrame(pred_train_dyn, columns=train_true.columns).set_index(train_true
    #                           [0:len(pred_train_dyn)].index)
    
        #for test set:
    
    #empty list for dynamic forecast values
    forecast_dyn=[]
    #make predictions over forecast horizon, then step over one forecast horizon
    for start in np.arange(0,len(test_target)-forecast_horizon,forecast_horizon):
        #only take relevant input data   
        input_temp_horizon=test_input[start: start+forecast_horizon].copy(deep=True)
        #empty list for forecast over one horizon    
        forecast_over_horizon=[]
        for step in range(0,forecast_horizon):
            #for first step, take the true data
            if step==0:
                input_temp = input_temp_horizon.iloc[step:step+1].copy()
            # for step 2 until end of forecast horizon: use predicted values 
            # of load with time lag 1-4
            else:
                input_temp_old=input_temp
                input_temp = input_temp_horizon.iloc[step:step+1].copy()
                input_temp.reset_index(drop=True,inplace=True)
                input_temp.loc[:,'lag1']=forecast_over_horizon[step-1]
                input_temp.loc[:,'lag2']=input_temp_old.loc[:,'lag1'].values
                input_temp.loc[:,'lag3']=input_temp_old.loc[:,'lag2'].values
                input_temp.loc[:,'lag4']=input_temp_old.loc[:,'lag3'].values
            #predict with relevant input data (only one time-step prediction per iteration)
            pred_step=model.predict(input_temp)
            forecast_over_horizon.append(pred_step)
    #append forecast over horizon to dynamic forecast
        forecast_over_horizon=np.ravel(forecast_over_horizon)
        forecast_dyn.append(forecast_over_horizon)
    #transform into one forecast float
    forecast_dyn=np.ravel(forecast_dyn)
    
    #re-scale
    pred_test_dyn=scaler.inverse_transform(forecast_dyn)
    #set time stamp and columnname
    # pred_test_dyn=pd.DataFrame(pred_test_dyn, columns=train_true.columns).set_index(test_true
    #                            [0:len(pred_test_dyn)].index)
      
    
    train_true=np.ravel(train_true)
    test_true=np.ravel(test_true)
    
    #defining RMSE
    RMSE = lambda real, predicted: np.sqrt(np.mean(np.power(real-predicted,2)))

    #getting static train and test scores
    score_stat_train=RMSE(train_true, pred_train)
    score_stat_test=RMSE(test_true, pred_test)
    score_dyn_train=RMSE(train_true[0:len(pred_train_dyn)], pred_train_dyn)
    score_dyn_test=RMSE(test_true[0:len(pred_test_dyn)], pred_test_dyn)
    
    score_stat_train=score_stat_train
    score_stat_test=score_stat_test
    score_dyn_train=score_dyn_train
    score_dyn_test=score_dyn_test
    return score_stat_train, score_stat_test, score_dyn_train, score_dyn_test


#test
#model=selected_model
#train_input=input_train_scaled
#train_target=target_train_scaled
#test_input=input_test_scaled
#test_target=target_test_scaled
#train_true=target_train_temp
#test_true=target_test_temp

#ann_prediction(selected_model,forecast_horizon, input_train_scaled, target_train_scaled, 
 #              input_test_scaled,target_test_scaled, target_train_temp, target_test_temp, scaler_Y)