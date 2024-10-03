# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:36:36 2020

@author: Erik Heilmann
"""

import pandas as pd
import numpy as np
import datasets
import converter
from forecast_methods_arimax import SklearnSarimax 
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn import neural_network
import itertools
from TimeShifter import TimeShift
from ANN_forecast import ann_prediction


## read in data
data = datasets.HessianLoadDataset("csv/",use_cache=False, data_scaler=None)

# put data in variable for vizualization    
#orinial_data = data.original_data    #all data
training_data=data.train_data
test_data=data.test_data
#validation_data=data.validation_data
    
# create TimeFeatures    'Month of Year', 'Day of Month' , 
                        #'Day of Week' and 'Hour of Day'

TFC=converter.TimeFeatureCreator(year=False,
                                         month_of_year = True,
                                         week_of_year = False,
                                         day_of_year = False,
                                         day_of_month = True,
                                         day_of_week = True,
                                         hour_of_day = True,
                                         minute_of_hour = False,
                                         do_sin_cos_encoding =True)


#write time features into DataFrames, name columns and set index
Time_train=pd.DataFrame(TFC.transform(data.train_data["inputs"][0].index)) #Time Features for train data
Time_train.columns=TFC.extracted_features_name # naming columns
Time_train.index = pd.DatetimeIndex(data.train_data["inputs"][0].index.values, freq='1H') # set index

Time_validation=pd.DataFrame(TFC.transform(data.validation_data["inputs"][0].index)) #time features for validationd ata
Time_validation.columns=TFC.extracted_features_name
Time_validation.index = pd.DatetimeIndex(data.validation_data["inputs"][0].index.values, freq='1H')

Time_test=pd.DataFrame(TFC.transform(data.test_data["inputs"][0].index)) #time features for test data
Time_test.columns=TFC.extracted_features_name
Time_test.index = pd.DatetimeIndex(data.test_data["inputs"][0].index.values, freq='1H')


# Complement Input-Data with Time-Stamp-Data for training, validation and test
# note, that the Input-Data are the same for whole Dataset

input_train=pd.concat([data.train_data["inputs"][0],Time_train], axis=1)
input_validation=pd.concat([data.validation_data["inputs"][0],Time_validation], axis=1)
input_test=pd.concat([data.test_data["inputs"][0],Time_test], axis=1)
                                
# define Reduced-Inputs for training, validation and test
# reduced Input-Data = Temperature, Hour of Day, Day of Week

reduced_features=['temp2m','day_of_week_sin', 'day_of_week_cos', 
                  'hour_of_day_sin', 'hour_of_day_cos' ]

input_train_reduced=input_train[reduced_features].copy()
input_validation_reduced=input_validation[reduced_features].copy()
input_test_reduced=input_test[reduced_features].copy()


# read in results
results_ANN_full=pd.read_pickle("stored_results_ANN_full.pkl")
results_ANN_full_trafo=pd.read_pickle("stored_results_ANN_full_transformer.pkl")
results_ANN_full=pd.concat([results_ANN_full,results_ANN_full_trafo],ignore_index=True)
results_ANN_reduced=pd.read_pickle("stored_results_ANN_reduced.pkl")
results_ANN_reduced_trafo=pd.read_pickle("stored_results_ANN_full_transformer.pkl")
results_ANN_reduced=pd.concat([results_ANN_reduced,results_ANN_reduced_trafo],ignore_index=True)

results_ARIMAX_full=pd.read_pickle("stored_results_ARIMAX_full.pkl")
results_ARIMAX_full_trafo=pd.read_pickle("stored_results_ARIMAX_full_transformer.pkl")
results_ARIMAX_full=pd.concat([results_ARIMAX_full,results_ARIMAX_full_trafo],ignore_index=True)
results_ARIMAX_reduced=pd.read_pickle("stored_results_ARIMAX_reduced.pkl")
results_ARIMAX_reduced_trafo=pd.read_pickle("stored_results_ARIMAX_full_transformer.pkl")
results_ARIMAX_reduced=pd.concat([results_ARIMAX_reduced,results_ARIMAX_reduced_trafo],ignore_index=True)

#example model recovering
forecast_horizon=24

## Define Data Format for Results (in same frame as primary results)                               

results=pd.DataFrame([],columns=['input_data', 'target', 'method', 'best_params', 
                                'train_score_static', 'train_score_dynamic',
                                'test_score_static', 'test_socre_dynamic', 
                                'forecast_horizon'])

##ANN
##full dataset
for i in range(0,len(results_ANN_full)): #
    
    name=results_ANN_full['target'][i]
    print(f"start iteration{1} with {name} in full dataset")

    #use fitted target 
    for k in range(0,len(data.train_data["names"])):
        if data.train_data["names"][k]==name:
            target_train=data.train_data["targets"][k]
            target_validation=data.validation_data["targets"][k]
            target_test=data.test_data["targets"][k]
    
    #pre-processing the data (using own scaler)
    
    #input pre-processing for ANN
    input_train_temp, target_train_temp=TimeShift(input_train, 
                                                  target_train)
    input_validation_temp, target_validation_temp=TimeShift(input_validation, 
                                                            target_validation)
    input_test_temp, target_test_temp=TimeShift(input_test, 
                                                  target_test)
    # normalization of all data
    scaler_Y=sklearn.preprocessing.StandardScaler()
    scaler_Y.fit(target_train_temp)
    scaler_X=sklearn.preprocessing.StandardScaler()
    scaler_X.fit(input_train_temp)
    
    target_train_scaled=scaler_Y.transform(target_train_temp)
    input_train_scaled=scaler_X.transform(input_train_temp)
    input_train_scaled=pd.DataFrame(input_train_scaled, columns=input_train_temp.columns)
    
    target_validation_scaled=scaler_Y.transform(target_validation_temp)
    input_validation_scaled=scaler_X.transform(input_validation_temp)
    input_validation_scaled=pd.DataFrame(input_validation_scaled, columns=input_validation_temp.columns)
    
    target_test_scaled=scaler_Y.transform(target_test_temp)
    input_test_scaled=scaler_X.transform(input_test_temp)
    input_test_scaled=pd.DataFrame(input_test_scaled, columns=input_test_temp.columns)


    #full input
    method='ANN' #for documentation
    input_data='full' #for documentation
    
    print(f"starting recalculation for {method} with {input_data} input data")
    
    ANN_params=results_ANN_full['best_params'][i]
    
    ANN=neural_network.MLPRegressor(max_iter=500, solver='adam', batch_size=100,
                                  validation_fraction=0.25, n_iter_no_change=20, 
                                  activation=ANN_params['activation'],
                                  alpha=ANN_params['alpha'],
                                  hidden_layer_sizes=ANN_params['hidden_layer_sizes'],
                                  learning_rate_init=ANN_params['learning_rate_init'])
    
    
    score_train, score_validation, score_dyn_train, score_dyn_validation = ann_prediction(ANN,
                                                                                  forecast_horizon, 
                                                                                  input_train_scaled,
                                                                                  target_train_scaled, 
                                                                                  input_test_scaled,
                                                                                  target_test_scaled,
                                                                                  target_train_temp,
                                                                                  target_test_temp, 
                                                                                  scaler_Y)
    
    results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                    'best_params':ANN_params, 'train_score_static':score_train, 
                                    'train_score_dynamic':score_dyn_train,'test_score_static':score_validation, 
                                    'test_socre_dynamic':score_dyn_validation, 
                                    'forecast_horizon':forecast_horizon}, ignore_index=True)
    
    print(f"finished recalculation for {method} with {input_data} input data")
    
    results.to_pickle(f"results/test_ANN_full.pkl")  
    print(f"saved {input_data} input data results of iteration {i}")


 ##reduced dataset   
for i in range(0,len(results_ANN_reduced)):
    
    name=results_ANN_reduced['target'][i]
    print(f"start iteration{1} with {name} in reduced dataset")

    #use fitted target 
    for k in range(0,len(data.train_data["names"])):
        if data.train_data["names"][k]==name:
            target_train=data.train_data["targets"][k]
            target_validation=data.validation_data["targets"][k]
            target_test=data.test_data["targets"][k]
    
    #pre-processing the data (using own scaler)
    
    #input pre-processing for ANN
    input_train_temp, target_train_temp=TimeShift(input_train_reduced, 
                                                  target_train)
    input_validation_temp, target_validation_temp=TimeShift(input_validation_reduced, 
                                                            target_validation)
    input_test_temp, target_test_temp=TimeShift(input_test_reduced, 
                                                  target_test)
    # normalization of all data
    scaler_Y=sklearn.preprocessing.StandardScaler()
    scaler_Y.fit(target_train_temp)
    scaler_X=sklearn.preprocessing.StandardScaler()
    scaler_X.fit(input_train_temp)
    
    target_train_scaled=scaler_Y.transform(target_train_temp)
    input_train_scaled=scaler_X.transform(input_train_temp)
    input_train_scaled=pd.DataFrame(input_train_scaled, columns=input_train_temp.columns)
    
    target_validation_scaled=scaler_Y.transform(target_validation_temp)
    input_validation_scaled=scaler_X.transform(input_validation_temp)
    input_validation_scaled=pd.DataFrame(input_validation_scaled, columns=input_validation_temp.columns)
    
    target_test_scaled=scaler_Y.transform(target_test_temp)
    input_test_scaled=scaler_X.transform(input_test_temp)
    input_test_scaled=pd.DataFrame(input_test_scaled, columns=input_test_temp.columns)


    #reduced input
    method='ANN' #for documentation
    input_data='reduced' #for documentation
    
    print(f"starting recalculation for {method} with {input_data} input data")
    
    ANN_params=results_ANN_reduced['best_params'][i]
    
    ANN=neural_network.MLPRegressor(max_iter=500, solver='adam', batch_size=100,
                                  validation_fraction=0.25, n_iter_no_change=20, 
                                  activation=ANN_params['activation'],
                                  alpha=ANN_params['alpha'],
                                  hidden_layer_sizes=ANN_params['hidden_layer_sizes'],
                                  learning_rate_init=ANN_params['learning_rate_init'])
    
    
    score_train, score_validation, score_dyn_train, score_dyn_validation = ann_prediction(ANN,
                                                                                  forecast_horizon, 
                                                                                  input_train_scaled,
                                                                                  target_train_scaled, 
                                                                                  input_test_scaled,
                                                                                  target_test_scaled,
                                                                                  target_train_temp,
                                                                                  target_test_temp, 
                                                                                  scaler_Y)
    

    
    results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                    'best_params':ANN_params, 
                                    'train_score_static':score_train, 
                                    'train_score_dynamic':score_dyn_train,
                                    'test_score_static':score_validation, 
                                    'test_socre_dynamic':score_dyn_validation, 
                                    'forecast_horizon':forecast_horizon}, ignore_index=True)
    
    print(f"finished recalculation for {method} with {input_data} input data")
    
    results.to_pickle(f"results/test_ANN_reduced.pkl")  
    print(f"saved {input_data} input data results of iteration {i}")










results=pd.DataFrame([],columns=['input_data', 'target', 'method', 'best_params', 
                                'train_score_static', 'train_score_dynamic',
                                'test_score_static', 'test_socre_dynamic', 
                                'forecast_horizon'])

    
    
    

##ARIMAX
##full dataset
for i in range(0,(len(results_ARIMAX_full)-1)): #
    
    name=results_ARIMAX_full['target'][i]
    print(f"start iteration{1} with {name} in full dataset")

    #use fitted target 
    for k in range(0,len(data.train_data["names"])):
        if data.train_data["names"][k]==name:
            target_train=data.train_data["targets"][k]
            target_validation=data.validation_data["targets"][k]
            target_test=data.test_data["targets"][k]
    

    #full input
    method='ARIMAX' #for documentation
    input_data='full' #for documentation
    
    print(f"starting recalculation for {method} with {input_data} input data")
    
    ARIMAX_params=results_ARIMAX_full['best_params'][i]
    
    # learn static model with optimal order           
    arimax=SklearnSarimax(order=ARIMAX_params['order'],
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
    arimax=SklearnSarimax(order=ARIMAX_params['order'],
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
    

    
    results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                    'best_params':ARIMAX_params, 'train_score_static':score_train, 
                                    'train_score_dynamic':score_dyn_train,'test_score_static':score_test, 
                                    'test_socre_dynamic':score_dyn_test, 
                                    'forecast_horizon':forecast_horizon}, ignore_index=True)
    
    print(f"finished recalculation for {method} with {input_data} input data")
    
    results.to_pickle(f"results/test_ARIMAX_full_plausibility.pkl")  
    print(f"saved {input_data} input data results of iteration {i}")


results=pd.DataFrame([],columns=['input_data', 'target', 'method', 'best_params', 
                                'train_score_static', 'train_score_dynamic',
                                'test_score_static', 'test_socre_dynamic', 
                                'forecast_horizon'])

 ##reduced dataset   
for i in range(0,(len(results_ARIMAX_reduced)-1)):
    
    name=results_ARIMAX_reduced['target'][i]
    print(f"start iteration{1} with {name} in reduced dataset")

    #use fitted target 
    for k in range(0,len(data.train_data["names"])):
        if data.train_data["names"][k]==name:
            target_train=data.train_data["targets"][k]
            target_validation=data.validation_data["targets"][k]
            target_test=data.test_data["targets"][k]
    
    

    #reduced input
    method='ARIMAX' #for documentation
    input_data='reduced' #for documentation
    
    print(f"starting recalculation for {method} with {input_data} input data")
    
    ARIMAX_params=results_ARIMAX_reduced['best_params'][i]
    
    # learn static model with optimal order           
    arimax=SklearnSarimax(order=ARIMAX_params['order'],
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
    arimax=SklearnSarimax(order=ARIMAX_params['order'],
                        dynamic=True,
                        use_exogenous_variables=True,
                        enforce_invertibility=False,
                        enforce_stationary=False,
                        forecast_horizon=forecast_horizon,
                        initialization="approximate_diffuse",
                        return_neg_score=False)
# fit on train data and get static score on train and test data
    arimax.fit(input_train_reduced, target_train)
    score_dyn_train=arimax.score(input_train_reduced, target_train, dynamic=True)
    score_dyn_test=arimax.score(input_test_reduced, target_test, dynamic=True)
    

    results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                    'best_params':ARIMAX_params, 'train_score_static':score_train, 
                                    'train_score_dynamic':score_dyn_train,'test_score_static':score_test, 
                                    'test_socre_dynamic':score_dyn_test, 
                                    'forecast_horizon':forecast_horizon}, ignore_index=True)
    

    
    print(f"finished recalculation for {method} with {input_data} input data")
    
    results.to_pickle(f"results/test_ARIMAX_reduced_plausibility.pkl")  
    print(f"saved {input_data} input data results of iteration {i}")









#read in residual data
residual_data =  pd.read_csv(f"residual_load.csv", sep=";", decimal=".",thousands=",", 
                             na_values=["-"], dtype={"Datum":str, "Uhrzeit":str}) 
residual_data.index = pd.to_datetime([f"{residual_data['Datum'][d]} {residual_data['Uhrzeit'][d]}" 
                                         for d in residual_data.index], format="%d.%m.%Y %H:%M:%S")
residual_data = residual_data.fillna(method="bfill")
residual_data = residual_data.drop(['Datum', 'Uhrzeit'],axis=1)

indices_train = pd.to_datetime(pd.date_range(start=data.train_data['targets'][0].index[0], end=data.train_data['targets'][0].index[-1], freq="H"))
train_residual = residual_data.loc[residual_data.index.intersection(indices_train)]['Residuallast']
train_residual=train_residual.resample('1H').asfreq()
train_residual=train_residual.interpolate()

indices_validation = pd.to_datetime(pd.date_range(start=data.validation_data['targets'][0].index[0], end=data.validation_data['targets'][0].index[-1], freq="H"))
validation_residual = residual_data.loc[residual_data.index.intersection(indices_validation)]['Residuallast']
validation_residual=validation_residual.resample('1H').asfreq()
validation_residual=validation_residual.interpolate()


indices_test = pd.to_datetime(pd.date_range(start=data.test_data['targets'][0].index[0], end=data.test_data['targets'][0].index[-1], freq="H"))
test_residual = residual_data.loc[residual_data.index.intersection(indices_test)]['Residuallast']

index_list=[]
for k in range(0,len(test_residual)):
    if str(test_residual.index[k]) in index_list:
        ind=test_residual.index[k]
        value=test_residual[k]
        test_residual=test_residual.drop(labels=[test_residual.index[k]])
    else:
        index_list.append(str(test_residual.index[k]))

test_residual=test_residual.resample('1H').asfreq()
test_residual=test_residual.interpolate()



results=pd.DataFrame([],columns=['input_data', 'target', 'method', 'best_params', 
                                'train_score_static', 'train_score_dynamic',
                                'test_score_static', 'test_socre_dynamic', 
                                'forecast_horizon'])

##ANN
##full dataset

    
name='residual_load'
print(f"start iteration{1} with {name} in full dataset")

#use fitted target 
target_train=train_residual
target_validation=validation_residual
target_test=test_residual

#pre-processing the data (using own scaler)

#input pre-processing for ANN
input_train_temp, target_train_temp=TimeShift(input_train, 
                                              target_train)
target_train_temp = target_train_temp.values.reshape(-1,1)
input_validation_temp, target_validation_temp=TimeShift(input_validation, 
                                                        target_validation)
target_validation_temp = target_validation_temp.values.reshape(-1,1)
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

target_validation_scaled=scaler_Y.transform(target_validation_temp)
input_validation_scaled=scaler_X.transform(input_validation_temp)
input_validation_scaled=pd.DataFrame(input_validation_scaled, columns=input_validation_temp.columns)

target_test_scaled=scaler_Y.transform(target_test_temp)
input_test_scaled=scaler_X.transform(input_test_temp)
input_test_scaled=pd.DataFrame(input_test_scaled, columns=input_test_temp.columns)


#full input
method='ANN' #for documentation
input_data='full' #for documentation

print(f"starting recalculation for {method} with {input_data} input data")

ANN_params=results_ANN_full_trafo['best_params'][0]

ANN=neural_network.MLPRegressor(max_iter=500, solver='adam', batch_size=100,
                              validation_fraction=0.25, n_iter_no_change=20, 
                              activation=ANN_params['activation'],
                              alpha=ANN_params['alpha'],
                              hidden_layer_sizes=ANN_params['hidden_layer_sizes'],
                              learning_rate_init=ANN_params['learning_rate_init'])


score_train, score_validation, score_dyn_train, score_dyn_validation = ann_prediction(ANN,
                                                                              forecast_horizon, 
                                                                              input_train_scaled,
                                                                              target_train_scaled, 
                                                                              input_test_scaled,
                                                                              target_test_scaled,
                                                                              target_train_temp,
                                                                              target_test_temp, 
                                                                              scaler_Y)

results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                'best_params':ANN_params, 'train_score_static':score_train, 
                                'train_score_dynamic':score_dyn_train,'test_score_static':score_validation, 
                                'test_socre_dynamic':score_dyn_validation, 
                                'forecast_horizon':forecast_horizon}, ignore_index=True)

print(f"finished recalculation for {method} with {input_data} input data")

results.to_pickle(f"results/test_ANN_residual.pkl")  
print(f"saved {input_data} input data results of iteration ")


 ##reduced dataset   
    
name='residual_load'
print(f"start iteration{1} with {name} in full dataset")

#use fitted target 
target_train=train_residual
target_validation=validation_residual
target_test=test_residual
    
#pre-processing the data (using own scaler)

#input pre-processing for ANN
input_train_temp, target_train_temp=TimeShift(input_train_reduced, 
                                              target_train)
target_train_temp = target_train_temp.values.reshape(-1,1)
input_validation_temp, target_validation_temp=TimeShift(input_validation_reduced, 
                                                        target_validation)
target_validation_temp = target_validation_temp.values.reshape(-1,1)
input_test_temp, target_test_temp=TimeShift(input_test_reduced, 
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

target_validation_scaled=scaler_Y.transform(target_validation_temp)
input_validation_scaled=scaler_X.transform(input_validation_temp)
input_validation_scaled=pd.DataFrame(input_validation_scaled, columns=input_validation_temp.columns)

target_test_scaled=scaler_Y.transform(target_test_temp)
input_test_scaled=scaler_X.transform(input_test_temp)
input_test_scaled=pd.DataFrame(input_test_scaled, columns=input_test_temp.columns)


#reduced input
method='ANN' #for documentation
input_data='reduced' #for documentation

print(f"starting recalculation for {method} with {input_data} input data")

ANN_params=results_ANN_reduced_trafo['best_params'][0]

ANN=neural_network.MLPRegressor(max_iter=500, solver='adam', batch_size=100,
                              validation_fraction=0.25, n_iter_no_change=20, 
                              activation=ANN_params['activation'],
                              alpha=ANN_params['alpha'],
                              hidden_layer_sizes=ANN_params['hidden_layer_sizes'],
                              learning_rate_init=ANN_params['learning_rate_init'])


score_train, score_validation, score_dyn_train, score_dyn_validation = ann_prediction(ANN,
                                                                              forecast_horizon, 
                                                                              input_train_scaled,
                                                                              target_train_scaled, 
                                                                              input_test_scaled,
                                                                              target_test_scaled,
                                                                              target_train_temp,
                                                                              target_test_temp, 
                                                                              scaler_Y)



results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                'best_params':ANN_params, 
                                'train_score_static':score_train, 
                                'train_score_dynamic':score_dyn_train,
                                'test_score_static':score_validation, 
                                'test_socre_dynamic':score_dyn_validation, 
                                'forecast_horizon':forecast_horizon}, ignore_index=True)

print(f"finished recalculation for {method} with {input_data} input data")

results.to_pickle(f"results/test_ANN_residual.pkl")  
print(f"saved {input_data} input data results of iteration")










results=pd.DataFrame([],columns=['input_data', 'target', 'method', 'best_params', 
                                'train_score_static', 'train_score_dynamic',
                                'test_score_static', 'test_socre_dynamic', 
                                'forecast_horizon'])

  

##ARIMAX
##full dataset

    
name='residual_load'
print(f"start iteration{1} with {name} in full dataset")

target_train=train_residual
target_validation=validation_residual
target_test=test_residual


#full input
method='ARIMAX' #for documentation
input_data='full' #for documentation

print(f"starting recalculation for {method} with {input_data} input data")

ARIMAX_params=results_ARIMAX_full_trafo['best_params'][0]

# learn static model with optimal order           
arimax=SklearnSarimax(order=ARIMAX_params['order'],
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
arimax=SklearnSarimax(order=ARIMAX_params['order'],
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



results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                'best_params':ARIMAX_params, 'train_score_static':score_train, 
                                'train_score_dynamic':score_dyn_train,'test_score_static':score_test, 
                                'test_socre_dynamic':score_dyn_test, 
                                'forecast_horizon':forecast_horizon}, ignore_index=True)

print(f"finished recalculation for {method} with {input_data} input data")

results.to_pickle(f"results/test_ARIMAX_full_trafo.pkl")  
print(f"saved {input_data} input data results of iteration ")



results=pd.DataFrame([],columns=['input_data', 'target', 'method', 'best_params', 
                                'train_score_static', 'train_score_dynamic',
                                'test_score_static', 'test_socre_dynamic', 
                                'forecast_horizon'])



 ##reduced dataset   

name='residual_load'
print(f"start iteration{1} with {name} in reduced dataset")

target_train=train_residual
target_validation=validation_residual
target_test=test_residual



#reduced input
method='ARIMAX' #for documentation
input_data='reduced' #for documentation

print(f"starting recalculation for {method} with {input_data} input data")

ARIMAX_params=results_ARIMAX_reduced_trafo['best_params'][0]

# learn static model with optimal order           
arimax=SklearnSarimax(order=ARIMAX_params['order'],
                    dynamic=False,
                    use_exogenous_variables=True,
                    enforce_invertibility=False,
                    enforce_stationary=False,
                    forecast_horizon=forecast_horizon,
                    initialization="approximate_diffuse",
                    return_neg_score=False)
# fit on train data and get static score on train and test data
arimax.fit(input_train_reduced, target_train)
score_train=arimax.score(input_train_reduced, target_train)
score_test=arimax.score(input_test_reduced, target_test)

# learn dynamic model with optimal order           
arimax=SklearnSarimax(order=ARIMAX_params['order'],
                    dynamic=True,
                    use_exogenous_variables=True,
                    enforce_invertibility=False,
                    enforce_stationary=False,
                    forecast_horizon=forecast_horizon,
                    initialization="approximate_diffuse",
                    return_neg_score=False)
# fit on train data and get static score on train and test data
arimax.fit(input_train_reduced, target_train)
score_dyn_train=arimax.score(input_train_reduced, target_train, dynamic=True)
score_dyn_test=arimax.score(input_test_reduced, target_test, dynamic=True)


results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                'best_params':ARIMAX_params, 'train_score_static':score_train, 
                                'train_score_dynamic':score_dyn_train,'test_score_static':score_test, 
                                'test_socre_dynamic':score_dyn_test, 
                                'forecast_horizon':forecast_horizon}, ignore_index=True)



print(f"finished recalculation for {method} with {input_data} input data")

results.to_pickle(f"results/test_ARIMAX_reduced_trafo.pkl")  
print(f"saved {input_data} input data results of iteration ")

























