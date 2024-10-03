# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:53:37 2020

@author: Erik Heilmann
"""
import pandas as pd
import numpy as np
import datasets
import converter
# from forecast_methods_arimax import SklearnSarimax 
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn import neural_network
import itertools
from TimeShifter import TimeShift
# from ANN_forecast import ann_prediction
import os
from experiment_functions import ARIMAX_experiment, ANN_experiment, check_nans

from logger import configure_logger
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from optparse import OptionParser, OptionGroup

import joblib

import sys


# create a slurm cluster to use 
#cluster = SLURMCluster(
#        queue="run",
#        project="econometrics",
#        cores=5,
#        name="econemtrics_worker",
#        local_directory="/tmp/",
#        log_directory="./logs/",
#        scheduler_options={"dashboard_address": "localhost:8787"},
#        env_extra = ['source /mnt/work/csells/paper-forecast-methods/env/bin/activate', 'cd /mnt/work/csells/paper-forecast-methods/load_forecast/'],
#        python = "python",
#        memory="5 GB")
#
#cluster.adapt(minimum_jobs=2,maximum_jobs=30)
#print("-----")
#print(cluster.job_script())
#print("-----")




def experiment(experiments_to_run,num_jobs=-1,use_only_transformer_data=False):
    # cluster = LocalCluster(n_workers=num_jobs, dashboard_address=None)

    # client = Client(cluster)


    

    ### READ IN DATA AND PRE-PROCESS
    data = datasets.HessianLoadDataset("csv",use_cache=False,nr_parallel_file_reader=num_jobs,data_scaler=None)

    # put data in variable for vizualization    
    #orinial_data = data.original_data    #all data
    training_data=data.train_data['targets'][0]
    test_data=data.test_data['targets'][0]
    validation_data=data.validation_data['targets'][0]
    
    
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

    indices_test = pd.to_datetime(pd.date_range(start=data.test_data['targets'][0].index[0], end=data.test_data['targets'][0].index[-1], freq="H"))
    test_residual= residual_data.loc[residual_data.index.intersection(indices_test)]['Residuallast']

        
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

                                
    ## Define Data Format for Results                               

    results=pd.DataFrame([],columns=['input_data', 'target', 'method', 'best_params', 
                                    'train_score', 'train_score_dynamic',
                                    'validation_score', 'validation_socre_dynamic', 
                                    'forecast_horizon'])


    # check nans for input data

    for i,input_data in enumerate([input_test, input_train, input_validation, input_train_reduced, input_validation_reduced, input_test_reduced]):
        if check_nans(input_data):
            print(f"{i}th input data contains nan")
            raise Exception("can't continue with nan in input set")

    #defining an forecast horizon (in h) for all dynamic prediction
    forecast_horizon=24 
                                    
    
    # define target dataset: 
        #option 1: for full dataset of targets use_only_transformer_data=False
        #option 2: for tranformer data use_only_transformer_data=True
    
    if use_only_transformer_data==False:
        data_len=len(data.train_data["names"])
    else:
        data_len=1

    
    #### Start Loop for Load-Datasets


    for i in range(0,data_len):
        
        if use_only_transformer_data==False:
            name=data.train_data["names"][i]
            target_train=data.train_data["targets"][i]
            target_validation=data.validation_data["targets"][i]
            target_test=data.test_data["targets"][i]
        else:
            name='residual_load'
            target_train=train_residual
            target_validation=validation_residual
            target_test=test_residual

     
        
        print(f"running {i}th loop dataset is now {name}")
        check_nans(target_train)
        check_nans(target_validation)
        check_nans(target_test)




        ### reduced Input-Data
        input_data='reduced' #for documentation
                        
        ## ARIMAX Optimization
        if ("ARIMAX_reduced" in experiments_to_run):
            method='ARIMAX' #for documentation
            
                                    
            score_train, score_validation, score_dyn_train, score_dyn_validation, best_params = ARIMAX_experiment(input_train_reduced, 
                                                                                                target_train, 
                                                                                                input_validation_reduced,
                                                                                                target_validation,
                                                                                                forecast_horizon, 
                                                                                                method, 
                                                                                                input_data,
                                                                                                n_jobs=num_jobs)
                                                                
            # Store Results                                   
            results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                    'best_params':best_params, 'train_score':score_train, 
                                    'train_score_dynamic':score_dyn_train,'validation_score':score_validation, 
                                    'validation_socre_dynamic':score_dyn_validation, 
                                    'forecast_horizon':forecast_horizon}, ignore_index=True)


        if ("ANN_reduced" in experiments_to_run):               
            # ANN Optimization
            method='ANN' #for documentation
            
            score_train, score_validation, score_dyn_train, score_dyn_validation, best_params = ANN_experiment(input_train_reduced, 
                                                                                                target_train, 
                                                                                                input_validation_reduced,
                                                                                                target_validation,
                                                                                                forecast_horizon,
                                                                                                method,
                                                                                                input_data,
                                                                                                n_jobs=num_jobs)

            # Store Results                                   
            results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                    'best_params':best_params, 'train_score':score_train, 
                                    'train_score_dynamic':score_dyn_train,'validation_score':score_validation, 
                                    'validation_socre_dynamic':score_dyn_validation, 
                                    'forecast_horizon':forecast_horizon}, ignore_index=True)
                                        
        ### full Input-Data
        input_data='full' #for documentation        

        if ("ARIMAX_full" in experiments_to_run):
            ## ARIMAX Optimization
            method='ARIMAX' #for documentation                             
                                                            
            score_train, score_validation, score_dyn_train, score_dyn_validation, best_params = ARIMAX_experiment(input_train, 
                                                                                                target_train, 
                                                                                                input_validation,
                                                                                                target_validation,
                                                                                                forecast_horizon,
                                                                                                method,
                                                                                                input_data,
                                                                                                n_jobs=num_jobs)
                                                                
            #   Store Results                                   
            results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                    'best_params':best_params, 'train_score':score_train, 
                                    'train_score_dynamic':score_dyn_train,'validation_score':score_validation, 
                                    'validation_socre_dynamic':score_dyn_validation, 
                                    'forecast_horizon':forecast_horizon}, ignore_index=True)


        if ("ANN_full" in experiments_to_run):
            ## ANN Optimization
            method='ANN' #for documentation

        
            score_train, score_validation, score_dyn_train, score_dyn_validation, best_params = ANN_experiment(input_train, 
                                                                                                target_train, 
                                                                                                input_validation,
                                                                                                target_validation,
                                                                                                forecast_horizon,
                                                                                                method,
                                                                                                input_data,
                                                                                                n_jobs=num_jobs)

            # Store Results                                   
            results=results.append({'input_data': input_data, 'target':name, 'method':method, 
                                    'best_params':best_params, 'train_score':score_train, 
                                    'train_score_dynamic':score_dyn_train,'validation_score':score_validation, 
                                    'validation_socre_dynamic':score_dyn_validation, 
                                    'forecast_horizon':forecast_horizon}, ignore_index=True)            
        
        used_transformer_data_str = "_transformer" if use_only_transformer_data else ""
        results.to_pickle(f"stored_results_{'_'.join(experiments)}{used_transformer_data_str}.pkl")  
        print(f"finished {i}th data loop for inputs of {name}")



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-j", "--jobs", dest="num_jobs", type="int",default=1)
    parser.add_option("--transformer", dest="use_only_transformer", action="store_const", default=False, const=True)

    methods = OptionGroup(parser,"Methods:","Can either be ANN or ARIMAX. Nothing is selected by default")
    methods.add_option("--ann",action="append_const", const="ANN", dest="methods")
    methods.add_option("--arimax",action="append_const", const="ARIMAX", dest="methods")
    parser.add_option_group(methods)

    data_partition = OptionGroup(parser,"DataPartitions:","Can either be full or reduced. Nothing is selected by default")
    data_partition.add_option("--full",action="append_const", const="full", dest="data_partition")
    data_partition.add_option("--reduced",action="append_const", const="reduced", dest="data_partition")
    parser.add_option_group(data_partition)
    
    # parser.add_option("-m", "--method", action="append", type="str", dest="methods",help="Choose the methods to run, can be ANN, ARIMAX, or 'ANN, ARIMAX'")
    # parser.add_option("-d", "--data", action="append", type="str", dest="datasize",help="Choose the data to run it on, can be full, reduced, or 'full,reduced'")
    configure_logger(True)
    (options, args) = parser.parse_args()
    print(options)
    
    if not options.methods:
        parser.error("no method given")
    if not options.data_partition:
        parser.error("no data partition given")

    
    experiments = ["_".join(x) for x in list(itertools.product(options.methods,options.data_partition))]

    num_jobs=options.num_jobs
    
    SLURM_CPUS_PER_TASK = os.getenv('SLURM_CPUS_PER_TASK')
    if SLURM_CPUS_PER_TASK is not None:
        # Set environment variables
        num_jobs=int(SLURM_CPUS_PER_TASK)

    experiment(experiments,num_jobs,use_only_transformer_data = options.use_only_transformer)