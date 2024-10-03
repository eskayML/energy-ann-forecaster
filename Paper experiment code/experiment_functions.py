# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:52:57 2020

@author: Erik Heilmann
"""
from forecast_methods_arimax import SklearnSarimax 
from TimeShifter import TimeShift
from sklearn.model_selection import GridSearchCV
import itertools
from ANN_forecast import ann_prediction
import sklearn
import pandas as pd
import numpy as np
from sklearn import neural_network

import logger as log

from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from optparse import OptionParser

import joblib
import warnings

def distribute_cv_fit(search_grid, input_data, target_data, random_seed=123):
    np.random.seed(random_seed)
    
    if type(target_data) == np.ndarray:
        target_1d = target_data.reshape(-1)
    else:
        target_1d = target_data.values.reshape(-1)
    if type(input_data) == np.ndarray:
        inputs = input_data
    else:
        inputs = input_data.values

    # with joblib.parallel_backend('dask',scatter=[inputs,target_1d]):
    
    search_grid.fit(inputs, target_1d)
    log.debug(__name__, "DONE with cv")
    
    return search_grid

            
def check_nans(df):
    result = False
    log.debug(__name__, f"df type is {type(df)}")
    if (type(df) == pd.DataFrame or type(df) == pd.Series):
        df = df.replace([np.inf, -np.inf], np.nan)
        result = df.isnull().any().any()
    elif type(df) == np.ndarray or type(df) == np.array:
        result = np.any(np.isnan(df)) or np.any(np.isinf(df))
    else: 
        log.warn(__name__, f"Wrong dtype: {type(df)}")
        result = True
    if (result):
        print("data is null/nan ")
        raise Exception("data is null/nan")
    else:
        return result

def ARIMAX_experiment(input_train_reduced, target_train, input_validation_reduced,
                   target_validation, forecast_horizon, method, input_data,n_jobs=-1):
       
                                
    # define model
        arimax=SklearnSarimax(dynamic=False,
                        use_exogenous_variables=True,
                        enforce_invertibility=False,
                        enforce_stationary=False,
                        forecast_horizon=forecast_horizon,
                        initialization="approximate_diffuse",
                        return_neg_score=True)

    # define hyperparameters
        p = q = range(0, 5) # p and q rank between 0 and 4
        d= range(0,3) # d rank between 0 and 2
        orders = list(itertools.product(p, d, q)) # combination of possible ranks
        
        #orders=[(1,1,1),(2,1,2),(3,1,3)]# testing order set


    # Start Grid Search

        
        search_grid=GridSearchCV(estimator=arimax, param_grid={'order': orders},n_jobs=n_jobs)


        print(f"starting {method} on {input_data}")
        search_grid = distribute_cv_fit(search_grid, input_train_reduced, target_train, random_seed=123)
        print(f"{method} complete")
        
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
        arimax.fit(input_train_reduced, target_train)
        score_train=arimax.score(input_train_reduced, target_train)
        score_validation=arimax.score(input_validation_reduced, target_validation)
        
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
        arimax.fit(input_train_reduced, target_train)
        score_dyn_train=arimax.score(input_train_reduced, target_train, dynamic=True)
        score_dyn_validation=arimax.score(input_validation_reduced, target_validation, dynamic=True)
        
        return (score_train, score_validation, score_dyn_train, score_dyn_validation, search_grid.best_params_)
        
def ANN_experiment(input_train_reduced, target_train, input_validation_reduced,
                   target_validation, forecast_horizon, method, input_data,n_jobs=-1):
    #Data pre-processing for ANN 
    # Time shifting (by lag of 4) target-data as inputs; 
    # only using complete data points (reduction of dataset by 4 datapoints)
        input_train_temp, target_train_temp=TimeShift(input_train_reduced, 
                                                    target_train)
        target_train_temp = target_train_temp.values.reshape(-1,1)
        input_validation_temp, target_validation_temp=TimeShift(input_validation_reduced, 
                                                                target_validation)
        target_validation_temp = target_validation_temp.values.reshape(-1,1)
        #input_test_temp, target_test_temp=TimeShift(input_test_reduced, 
        #                                            target_test)
        # target_test_temp = target_test_temp.values.reshape(-1,1)

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

        #target_test_scaled=scaler_Y.transform(target_test_temp)
        #input_test_scaled=scaler_X.transform(input_test_temp)
        #input_test_scaled=pd.DataFrame(input_test_scaled, columns=input_test_temp.columns)

                                
    # define model
        ANN=neural_network.MLPRegressor(solver='adam', batch_size=100,
                                    validation_fraction=0.25, n_iter_no_change=20)
    # define hyperparameters
        alphas= 10.0 ** -np.arange(1, 4)
        # alphas=[0.1,0.01]#for testing
        
        learning_rate_init=[0.01, 0.001, 0.0001]
        # learning_rate_init=[0.001] # for testing
        
        layers=[] #combine different network architectures with 3 Layers
        units=[5,10,15]
        for j in units:
            for k in units:
                layers.append((j,k))
                for l in units:
                    layers.append((j,k,l))
        # layers=[(10,10,10),(50,50,50)]  # for testing

        activation=['tanh', 'relu']              
        # activation=['relu']      #for testing
        
    # Start Grid Search
        
        search_grid=GridSearchCV(estimator=ANN, param_grid={'alpha': alphas,
                                                            'learning_rate_init': learning_rate_init,
                                                            'hidden_layer_sizes':layers, 
                                                            'activation':activation},
                                    verbose=2,
                                    n_jobs=n_jobs
            )
        check_nans(input_train_scaled)
        check_nans(target_train_scaled)
        print(f"starting {method} on {input_data}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search_grid = distribute_cv_fit(search_grid, input_train_scaled, target_train_scaled, random_seed=123)
        print(f"{method} complete")
        

    #making predictions and getting score
        selected_model=search_grid.best_estimator_
        


        score_train, score_validation, score_dyn_train, score_dyn_validation = ann_prediction(selected_model,
                                                                                forecast_horizon, 
                                                                                input_train_scaled,
                                                                                target_train_scaled, 
                                                                                input_validation_scaled,
                                                                                target_validation_scaled,
                                                                                target_train_temp,
                                                                                target_validation_temp, 
                                                                                scaler_Y)        
        
        return (score_train, score_validation, score_dyn_train, score_dyn_validation, search_grid.best_params_)
   
    
    
        