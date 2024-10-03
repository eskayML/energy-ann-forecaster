# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:36:36 2020

@author: Erik Heilmann
"""

import pandas as pd
import numpy as np
import converter
from forecast_methods_arimax import SklearnSarimax 
from sklearn.model_selection import GridSearchCV
from sklearn import neural_network
import itertools
from TimeShifter import TimeShift
from ANN_forecast import ann_prediction
from forecast_methods_arimax import SklearnSarimax 
from ANN_forecast import ann_prediction
import sklearn
import warnings


'''
The code below contains an easy example for the application of the machine learning modeling prcoess.
The target dataset is an exemplary load dataset.
The input dataset only contains the time stamp of the load data.
The application contains an ARIMAX model and an ANN model.
Therefore, a short grid search is implemented for both apporaches.
However, due to the simplification of the input datset as well as an shortened grid search,
the resulting models should not be interpreted as well performing, but only an exemplary application.

The proceed follows the machine learning process introduced in Section 3.1 of the paper.
'''


'''
Steps 1 and 2: read in data and divide into train, validation and test data.
'''

## read in exemplary data and adjust data frame
   
exemplary_data =  pd.read_csv("exemplary_load.csv", sep=";", decimal=".",thousands=",", 
                             na_values=["-"], dtype={"Datum":str, "Uhrzeit":str}) 

# # EXP: truncating the dataset
# exemplary_data = exemplary_data[:1000]



exemplary_data.index = pd.to_datetime([f"{exemplary_data['Datum'][d]} {exemplary_data['Uhrzeit'][d]}" 
                                         for d in exemplary_data.index], format="%d.%m.%Y %H:%M:%S")
exemplary_data = exemplary_data.fillna(method="bfill")
exemplary_data = exemplary_data.drop(['Datum', 'Uhrzeit'],axis=1)

## define train, validation and target dataset
target_train=exemplary_data[:round(0.8*len(exemplary_data))]
#Note, that the train set contains train data (60% of dataset) and vvalidation data (20% of dataset)
#the implemented grid search function split train and validation data on its own
#target_validation=exemplary_data[round(0.6*len(exemplary_data)):round(0.8*len(exemplary_data))]
target_test=exemplary_data[round(0.8*len(exemplary_data)):]

##crate exemplary input dataset: TimeFeatures 'Month of Year', 'Day of Month' , 'Day of Week' and 'Hour of Day'

TFC=converter.TimeFeatureCreator(year=False,
                                         month_of_year = True,
                                         week_of_year = False,
                                         day_of_year = False,
                                         day_of_month = True,
                                         day_of_week = True,
                                         hour_of_day = True,
                                         minute_of_hour = False,
                                         do_sin_cos_encoding =True)

#write time features into Input-DataFrames, name columns and set index
input_train=pd.DataFrame(TFC.transform(target_train.index)) #Time Features for train data
input_train.columns=TFC.extracted_features_name # naming columns
input_train.index = pd.DatetimeIndex(target_train.index.values) # set index
#train data contain validation data in this example!

#input_validation=pd.DataFrame(TFC.transform(target_validation.index)) #time features for validationd ata
#input_validation.columns=TFC.extracted_features_name
#input_validation.index = pd.DatetimeIndex(target_validation.index.values)

input_test=pd.DataFrame(TFC.transform(target_test.index)) #time features for test data
input_test.columns=TFC.extracted_features_name
input_test.index = pd.DatetimeIndex(target_test.index.values)
input_data='Only_time_params' #for documentation


#define forecast horizon for dynamic forecast
forecast_horizon=24

## Define Data Format for Results (in same frame as primary results)                               
results = pd.DataFrame(columns=['input_data', 'target', 'method', 'best_params', 
                                'train_score_static', 'train_score_dynamic',
                                'test_score_static', 'test_socre_dynamic', 
                                'forecast_horizon'])
'''
STEP 3a: Select Model approach.
Option 1 - ARIMAX MODEL
'''

method='ARIMAX' #for documentation

arimax=SklearnSarimax(dynamic=False,
                use_exogenous_variables=True,
                enforce_invertibility=False,
                enforce_stationary=False,
                forecast_horizon=forecast_horizon,
                initialization="approximate_diffuse",
                return_neg_score=True)

'''
Step 3b: Define a grid search for the model apporach
'''

# define hyperparameters

# larger grid search:
#p = q = range(0, 4) # p and q rank between 0 and 3
#d= range(0,2) # d rank between 0 and 1
#orders = list(itertools.product(p, d, q)) # combination of possible ranks

#small grid search:

orders=[(1,1,1),(2,1,2),(3,1,3)]
# orders = [(1,1,1), (0,1,0), (1,1,1)]
'''
Steps 4a - 4c Model selection via grid search 
'''
# Start Grid Search
search_grid=GridSearchCV(estimator=arimax, param_grid={'order': orders},n_jobs=1)
print(f"starting grid search of {method} on {input_data}")
np.random.seed(123)
if type(target_train) == np.ndarray:
    target_1d = target_train.reshape(-1)
else:
    target_1d = target_train.values.reshape(-1)
if type(input_train) == np.ndarray:
    inputs = input_train
else:
    inputs = input_train.values
search_grid.fit(inputs, target_1d)
print(f"grid search of {method} complete")


'''
Steps 5 and 6: select best the best model of the grid search and evaluate the performance of the model.
Note that, for documentation, the model performance is evaluated on the test dataset as well as on the training data.
In addition, a static forecast as well as an dynamic forecast over the defined forecast horizon is evaluated.
'''

# learn static model with optimal order           
arimax=SklearnSarimax(order=search_grid.best_params_['order'],
                    dynamic=False,
                    use_exogenous_variables=True,
                    enforce_invertibility=False,
                    enforce_stationary=False,
                    forecast_horizon=forecast_horizon,
                    initialization="approximate_diffuse",
                    return_neg_score=False)
# fit on train data and get static score on train and test data
arimax.fit(input_train, target_train)




score_train=arimax.score(input_train, target_train)
score_test=arimax.score(input_test, target_test)

# learn dynamic model with optimal order           
arimax=SklearnSarimax(order=search_grid.best_params_['order'],
                    dynamic=True,
                    use_exogenous_variables=True,
                    enforce_invertibility=False,
                    enforce_stationary=False,
                    forecast_horizon=forecast_horizon,
                    initialization="approximate_diffuse",
                    return_neg_score=False)
# fit on train data and get static score on train and test data
arimax.fit(input_train, target_train)
score_dyn_train=arimax.score(input_train, target_train, dynamic=True)
score_dyn_test=arimax.score(input_test, target_test, dynamic=True)

results=pd.concat([results,pd.DataFrame({'input_data': input_data, 'target':'exemplary_data', 'method':method, 
                        'best_params':search_grid.best_params_, 'train_score_static':score_train, 
                        'train_score_dynamic':score_dyn_train,'test_score_static':score_test, 
                        'test_socre_dynamic':score_dyn_test, 
                        'forecast_horizon':forecast_horizon})], ignore_index=True)




    
'''
STEP 3a: Select Model approach.
Option 2 - ANN MODEL

Note that the application of ANN needs additional pre-processing steps.
First, the input data get complemented with the past 4 values of the target dataset.
Second all data (input and tagret) get scaled.
'''
method='ANN' #for documentation
            
#Data pre-processing for ANN 
# Time shifting (by lag of 4) target-data as inputs; 
# only using complete data points (reduction of dataset by 4 datapoints)
input_train_temp, target_train_temp=TimeShift(input_train, 
                                            target_train)
target_train_temp = target_train_temp.values.reshape(-1,1)


input_test_temp, target_test_temp=TimeShift(input_test, 
                                            target_test)
target_test_temp = target_test_temp.values.reshape(-1,1)

# normalization of all data
scaler_Y=sklearn.preprocessing.StandardScaler()
scaler_Y.fit(target_train_temp)
scaler_X=sklearn.preprocessing.StandardScaler()
scaler_X.fit(input_train_temp)

target_train_scaled=scaler_Y.transform(target_train_temp)
input_train_scaled=scaler_X.transform(input_train_temp)
input_train_scaled=pd.DataFrame(input_train_scaled, columns=input_train_temp.columns)

target_test_scaled=scaler_Y.transform(target_test_temp)
input_test_scaled=scaler_X.transform(input_test_temp)
input_test_scaled=pd.DataFrame(input_test_scaled, columns=input_test_temp.columns)

                        
# define model
ANN=neural_network.MLPRegressor(solver='adam', batch_size=100,
 
                                validation_fraction=0.25, n_iter_no_change=20)


'''
Step 3b: Define a grid search for the model apporach
'''
# define hyperparameters for grid search
#alphas= 10.0 ** -np.arange(1, 4) #larger grid search
alphas=[0.1,0.01]

#learning_rate_init=[0.01, 0.001, 0.0001] #larger grid search
learning_rate_init=[0.001] 

#layers=[] # larger grid search combine different network architectures with 3 Layers
#units=[5,10,15]
#for j in units:
#    for k in units:
#        layers.append((j,k))
#        for l in units:
#            layers.append((j,k,l))
layers=[(10,10,10),(50,50,50)] 

#activation=['tanh', 'relu']#larger grid search              
activation=['relu']      



'''
Steps 4a - 4c Model selection via grid search 
'''
# Start Grid Search
search_grid=GridSearchCV(estimator=ANN, param_grid={'alpha': alphas,
                                                    'learning_rate_init': learning_rate_init,
                                                    'hidden_layer_sizes':layers, 
                                                    'activation':activation},
                            verbose=2,
                            n_jobs=1
    )

print(f"starting grid search for {method} on {input_data}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
 
    np.random.seed(123)
    
    if type(target_train_scaled) == np.ndarray:
        target_1d = target_train_scaled.reshape(-1)
    else:
        target_1d = target_train_scaled.values.reshape(-1)
    if type(input_train_scaled) == np.ndarray:
        inputs = input_train_scaled
    else:
        inputs = input_train_scaled.values

    
    search_grid.fit(inputs, target_1d)

print(f"Grid search for {method} complete")


'''
Steps 5 and 6: select best the best model of the grid search and evaluate the performance of the model.
Note that, for documentation, the model performance is evaluated on the test dataset as well as on the training data.
In addition, a static forecast as well as an dynamic forecast over the defined forecast horizon is evaluated.
'''

#select best model
selected_model=search_grid.best_estimator_
#fit selected model
np.random.seed(123)
selected_model.fit(input_train_scaled, target_train_scaled)

##static prediciton

#for training set
pred_train_scaled = selected_model.predict(input_train_scaled)
#reshape to 2D array
pred_train_scaled = pred_train_scaled.reshape(-1, 1)
#re-scale
pred_train = scaler_Y.inverse_transform(pred_train_scaled)

#set time stamp and columnname
# pred_train=pd.DataFrame(pred_train).set_index(train_true.index)

#test set
pred_test_scaled = selected_model.predict(input_test_scaled)
#reshape to 2D array
pred_test_scaled = pred_test_scaled.reshape(-1, 1)
#re-scale
pred_test = scaler_Y.inverse_transform(pred_test_scaled)
##dynamic prediction

#for train set:

#empty list for dynamic forecast values
forecast_dyn=[]
#make predictions over forecast horizon, then step over one forecast horizon
for start in np.arange(0,len(target_train_scaled)-forecast_horizon,forecast_horizon):
    #only take relevant input data   
    input_temp_horizon=input_train_scaled[start: start+forecast_horizon].copy(deep=True)
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
        pred_step=selected_model.predict(input_temp)
        forecast_over_horizon.append(pred_step)
#append forecast over horizon to dynamic forecast
    forecast_over_horizon=np.ravel(forecast_over_horizon)
    forecast_dyn.append(forecast_over_horizon)
#transform into one forecast float
forecast_dyn=np.ravel(forecast_dyn)
forecast_dyn = forecast_dyn.reshape(-1,1)
#re-scale
pred_train_dyn=scaler_Y.inverse_transform(forecast_dyn)
#set time stamp and columnname
#pred_train_dyn=pd.DataFrame(pred_train_dyn, columns=train_true.columns).set_index(train_true
#                           [0:len(pred_train_dyn)].index)

    #for test set:

#empty list for dynamic forecast values
forecast_dyn=[]
#make predictions over forecast horizon, then step over one forecast horizon
for start in np.arange(0,len(target_test_scaled)-forecast_horizon,forecast_horizon):
    #only take relevant input data   
    input_temp_horizon=input_test_scaled[start: start+forecast_horizon].copy(deep=True)
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
        pred_step=selected_model.predict(input_temp)
        forecast_over_horizon.append(pred_step)
#append forecast over horizon to dynamic forecast
    forecast_over_horizon=np.ravel(forecast_over_horizon)
    forecast_dyn.append(forecast_over_horizon)
#transform into one forecast float
forecast_dyn=np.ravel(forecast_dyn)

#re-scale
pred_test_dyn=scaler_Y.inverse_transform(forecast_dyn.reshape(-1,1))
#set time stamp and columnname
# pred_test_dyn=pd.DataFrame(pred_test_dyn, columns=train_true.columns).set_index(test_true
#                            [0:len(pred_test_dyn)].index)
  

train_true=np.ravel(target_train_temp)
test_true=np.ravel(target_test_temp)

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
    
            
results=pd.concat([results,pd.DataFrame({'input_data': input_data, 'target':'exemplary_data', 'method':method, 
                        'best_params':search_grid.best_params_, 'train_score_static':score_train, 
                        'train_score_dynamic':score_dyn_train,'test_score_static':score_test, 
                        'test_socre_dynamic':score_dyn_test, 
                        'forecast_horizon':forecast_horizon})], ignore_index=True)
            
            

results.to_csv("example.csv")  
print("Example finished.")

























