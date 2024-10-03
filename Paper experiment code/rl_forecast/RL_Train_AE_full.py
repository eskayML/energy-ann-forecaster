#!/usr/bin/env python
# coding: utf-8

# * [x] Load Data
# * [x] Create CNN 
# * [x] Train CNN
# * [ ] Visualize CNN
# * [ ] Train Forecast using CNN
# * [ ] Visualize Forecast CNN

# In[1]:


import pandas as pd
import numpy as np
import torch
import copy
import itertools
import os

import datasets, converter

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.callbacks import OneCycleScheduler, EarlyStoppingCallback
from fastai.basic_data import DatasetType
from fastai.metrics import root_mean_squared_error
from util import convert_ts_data_to_cnn_ts_data,FastAICompatibleDataSet, dev_to_np
from autoencoder import Autoencoder, AutoLSTM
from functools import partial
from sklearn.preprocessing import MinMaxScaler

from optparse import OptionParser, OptionGroup





def init_model(m,init_function=torch.nn.init.xavier_normal_):
    if type(m) == torch.nn.Linear:
        init_function(m.weight)
        torch.nn.init.normal_(m.bias.data)
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
        init_function(m.weight.data)
        m.bias.data.fill_(0.1)
    if (isinstance(m,torch.nn.LSTM)):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)



def check_nans(df):
    result = False
    if (type(df) == pd.DataFrame or type(df) == pd.Series):
        df = df.replace([np.inf, -np.inf], np.nan)
        result = df.isnull().any().any()
    elif type(df) == np.ndarray or type(df) == np.array:
        result = np.any(np.isnan(df)) or np.any(np.isinf(df))
    else: 
        result = True
    if (result):
        print("data is null/nan ")
        raise Exception("data is null/nan")
    else:
        return result


def transform_to_sequence(data,timesteps=24):
    samples = data.shape[0]
    features = data.shape[1]
    return data[range(samples//timesteps * timesteps),:].reshape(-1,timesteps,features)

def experiment(config):

    # PARAMETER!
    torch.manual_seed(42)
    np.random.seed(42)


    batch_size = config.batch_size
    max_epoch = config.epochs

    lr = config.lr
    lr_max = config.lr_max

    shuffle = config.shuffle_data
    drop_last = config.drop_last

    splits = [0.7,0.2,0.1]


    num_jobs=config.num_jobs

    device = "cpu"

    plot = False
    reduced = config.reduced

    # read in data
    data = datasets.HessianLoadDataset("../csv",use_cache=False,nr_parallel_file_reader=num_jobs,data_scaler=None)

    # put data in variable for vizualization    
    #orinial_data = data.original_data    #all data



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

    cos_sin_header = [x for x in Time_train.columns if ('sin' in x or 'cos' in x)]
    Time_train = Time_train[cos_sin_header]
    Time_validation = Time_validation[cos_sin_header]
    Time_test = Time_test[cos_sin_header]

    # Complement Input-Data with Time-Stamp-Data for training, validation and test
    # note, that the Input-Data are the same for whole Dataset

    input_train=pd.concat([data.train_data["inputs"][0],Time_train], axis=1)
    input_validation=pd.concat([data.validation_data["inputs"][0],Time_validation], axis=1)
    input_test=pd.concat([data.test_data["inputs"][0],Time_test], axis=1)

    # recombine to perform random splits
    if config.random_val_train_test:
        old_input_train = input_train.copy()
        old_input_validation = input_validation.copy()
        old_input_test = input_test.copy()

        all_input_data = pd.concat([input_train,input_validation,input_test])
        
        max_ts_len = len(all_input_data.index)//24*24
        ix = np.arange(0,max_ts_len)
        ix_split = ix.reshape(-1,24)
        ix_split.shape[0]

        perm = np.random.permutation(np.arange(0,ix_split.shape[0]))
        
        train_start_ix = 0
        train_end_ix = int(ix_split.shape[0]*splits[0])
        val_start_ix = int(ix_split.shape[0]*splits[0]+1)
        val_end_ix = int(ix_split.shape[0]*(splits[1]+splits[0]))
        test_start_ix = int(ix_split.shape[0]*(splits[1]+splits[0])+1)
        test_end_ix = ix_split.shape[0]

        train_ix = ix_split[perm[train_start_ix:train_end_ix]].flatten()
        input_train = all_input_data.iloc[train_ix]

        val_ix = ix_split[perm[val_start_ix:val_end_ix]].flatten()
        input_validation = all_input_data.iloc[val_ix]

        test_ix = ix_split[perm[test_start_ix:test_end_ix]].flatten()
        input_test = all_input_data.iloc[test_ix]


        indices_train = all_input_data.index[train_ix]

        train_residual = residual_data.loc[residual_data.index.intersection(indices_train)]['Residuallast']
        train_residual = train_residual.drop_duplicates()
        train_residual=train_residual.resample('1H').asfreq()
        train_residual=train_residual.interpolate()
    
        indices_validation = all_input_data.index[val_ix]
        validation_residual = residual_data.loc[residual_data.index.intersection(indices_validation)]['Residuallast']
        validation_residual = validation_residual.drop_duplicates()
        validation_residual=validation_residual.resample('1H').asfreq()
        validation_residual=validation_residual.interpolate()

        indices_test = all_input_data.index[test_ix]
        test_residual= residual_data.loc[residual_data.index.intersection(indices_test)]['Residuallast']
        test_residual = test_residual.drop_duplicates()
        test_residual=test_residual.resample('1H').asfreq()
        test_residual=test_residual.interpolate()

    # define Reduced-Inputs for training, validation and test
    # reduced Input-Data = Temperature, Hour of Day, Day of Week


    reduced_features=['temp2m','day_of_week_sin', 'day_of_week_cos', 
                    'hour_of_day_sin', 'hour_of_day_cos' ]

    input_train_reduced=input_train[reduced_features].copy()
    input_validation_reduced=input_validation[reduced_features].copy()
    input_test_reduced=input_test[reduced_features].copy()


    for x in [input_test,input_validation,input_train]:
        x.drop(['preice'],inplace=True, axis=1)

    # preprocess data

    input_scaler= MinMaxScaler()
    input_scaler.fit(input_train)
    input_train_scaled = input_scaler.transform(input_train)
    input_validation_scaled = input_scaler.transform(input_validation)
    input_test_scaled = input_scaler.transform(input_test)


    input_scaler_reduced= MinMaxScaler()
    input_train_reduced=input_train[reduced_features].copy()
    input_scaler_reduced.fit(input_train_reduced)
    input_train_reduced = input_scaler_reduced.transform(input_train_reduced)
    input_validation_reduced=input_scaler_reduced.transform(input_validation[reduced_features].copy())
    input_test_reduced=input_scaler_reduced.transform(input_test[reduced_features].copy())
    
    

    print(f"Reduced shape{input_train_reduced.shape}")
    print(f"Non Reduced shape{input_train_scaled.shape}")

    for i,input_data in enumerate([input_test, input_train, input_validation, input_train_reduced, input_validation_reduced, input_test_reduced,input_train_scaled,input_validation_scaled,input_test_scaled]):
        print(i)
        if check_nans(input_data):
            print(f"{i}th input data contains nan")
            raise Exception("can't continue with nan in input set")

    
        
    input_train_scaled_seq = transform_to_sequence(input_train_scaled)
    input_validation_scaled_seq = transform_to_sequence(input_validation_scaled)
    input_test_scaled_seq = transform_to_sequence(input_test_scaled)

    train_dl = DataLoader(FastAICompatibleDataSet(input_train_scaled,input_train_scaled),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
    val_dl = DataLoader(FastAICompatibleDataSet(input_validation_scaled,input_validation_scaled),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
    test_dl = DataLoader(FastAICompatibleDataSet(input_test_scaled,input_test_scaled),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
    if (reduced):
        train_dl = DataLoader(FastAICompatibleDataSet(input_train_reduced,input_train_reduced),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
        val_dl = DataLoader(FastAICompatibleDataSet(input_validation_reduced,input_validation_reduced),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
        test_dl = DataLoader(FastAICompatibleDataSet(input_test_reduced,input_test_reduced),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)

    db = DataBunch(train_dl, val_dl, test_dl=test_dl)

    # TRAIN AE

    loss = torch.nn.MSELoss()

    # define autoencoder
    if config.load_ae_model:
        model = torch.load(f"models/autoencoder{'_reduced' if config.reduced else ''}.model")

    else:
        model = Autoencoder(input_train_scaled.shape[1],[30,20, 10]).to(device)
        if reduced:
            model = Autoencoder(input_train_reduced.shape[1],[10, 7, 4]).to(device)

        model.apply(lambda m: init_model(m))

        #Early stoppping if neededpartial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.001, patience=20)
        learner = Learner(db,model,loss_func=loss, callback_fns=[partial(OneCycleScheduler,lr_max=lr_max)] )

        learner.fit(max_epoch,lr = lr)

        if reduced:
            torch.save(model, "models/autoencoder_reduced.model")
        else:
            torch.save(model, "models/autoencoder.model")
    

        y_pred,y_true = learner.get_preds(DatasetType.Valid)
        y_pred = dev_to_np(y_pred)
        y_true = dev_to_np(y_true)

        print(f"AE RMSE on validation: {np.sqrt(np.mean(np.power(y_true-y_pred,2)))}")





    # fc_train_dl = DataLoader(FastAICompatibleDataSet(input_train_scaled,target_scaler.transform(training_data)),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
    # fc_val_dl = DataLoader(FastAICompatibleDataSet(input_validation_scaled,target_scaler.transform(validation_data)),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
    # fc_test_dl = DataLoader(FastAICompatibleDataSet(input_test_scaled,target_scaler.transform(test_data)),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)

    # if (reduced):
    #     fc_train_dl = DataLoader(FastAICompatibleDataSet(input_train_reduced,target_scaler.transform(training_data)),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
    #     fc_val_dl = DataLoader(FastAICompatibleDataSet(input_validation_reduced,target_scaler.transform(validation_data)),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
    #     fc_test_dl = DataLoader(FastAICompatibleDataSet(input_test_reduced,target_scaler.transform(test_data)),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
    results=pd.DataFrame([],columns=['input_data', 'target', 'method', 'best_params', 
                                        'train_score', 'train_score_dynamic',
                                        'validation_score', 'validation_socre_dynamic', "test_score",
                                        'forecast_horizon'])

    train_length = len(data.train_data['targets'])
    if config.residual:
        train_length = 1
    for i in range(train_length):

        
        cur_name = data.train_data['names'][i]

        if config.skip_trained and os.path.isfile(f"load_forecast/rl_forecast/models/autolstm_{cur_name}.model"):
            continue

        if config.skip_residual_distributed and 'Load1' in cur_name:
            continue

        
        training_data=data.train_data['targets'][i]
        validation_data=data.validation_data['targets'][i]
        test_data=data.test_data['targets'][i]

        if config.random_val_train_test:
            all_target_data = pd.concat([training_data,validation_data,test_data])
            training_data = all_target_data.iloc[train_ix]
            validation_data = all_target_data.iloc[val_ix]
            test_data = all_target_data.iloc[test_ix]

        target_scaler= MinMaxScaler()
        target_scaler.fit(training_data)



        if reduced:
            input_train_scaled_seq = transform_to_sequence(input_train_reduced)
            input_validation_scaled_seq = transform_to_sequence(input_validation_reduced)
            input_test_scaled_seq = transform_to_sequence(input_test_reduced)
        
        
        
            

        target_train_seq = transform_to_sequence(target_scaler.transform(training_data))
        target_val_seq = transform_to_sequence(target_scaler.transform(validation_data))
        target_test_seq = transform_to_sequence(target_scaler.transform(test_data))
        if config.residual and train_length == 1:
            cur_name = "residual"
            target_scaler= MinMaxScaler()
            target_scaler.fit(train_residual.values.reshape(-1,1))
            target_train_seq = transform_to_sequence(target_scaler.transform(train_residual.values.reshape(-1,1)))
            target_val_seq = transform_to_sequence(target_scaler.transform(validation_residual.values.reshape(-1,1)))
            target_test_seq = transform_to_sequence(target_scaler.transform(test_residual.values.reshape(-1,1)))
            

        fc_train_dl_seq = DataLoader(FastAICompatibleDataSet(input_train_scaled_seq,target_train_seq),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
        fc_val_dl_seq = DataLoader(FastAICompatibleDataSet(input_validation_scaled_seq,target_val_seq),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)
        fc_test_dl_seq = DataLoader(FastAICompatibleDataSet(input_test_scaled_seq,target_test_seq),batch_size, shuffle=shuffle,drop_last=drop_last, pin_memory=True if device == "cpu" else False)


        # db = DataBunch(train_dl, val_dl, test_dl=test_dl)
        # fc_db = DataBunch(fc_train_dl, fc_val_dl, test_dl=fc_test_dl)
        fc_seq_db = DataBunch(fc_train_dl_seq, fc_val_dl_seq, test_dl=fc_test_dl_seq)


        best_model = None
        best_score = {"validation": 1000000000000000.0}
        best_params = None


        nr_lstm_stages = [1,2,4]
        hidden_sizes = [1,5,10]
        levels_to_retrain = [1]
        hidden_states_init_random = [True, False]
        
        grid_search_params = itertools.product(nr_lstm_stages,hidden_sizes,levels_to_retrain,hidden_states_init_random)
        print(f"starting grid search on {cur_name}")
        for params in grid_search_params:
            lstm_stages = params[0]
            hidden_size = params[1]
            levels_to_retrain = params[2]
            hidden_states_zero = params[3]

            # auto lstm model

            autolstm = AutoLSTM(model,lstm_stages,hidden_size=hidden_size, levels_to_retrain=levels_to_retrain, random_hidden_states=hidden_states_zero)


            autolstm.apply(lambda m: init_model(m))

            # Early stopping if needed,partial(EarlyStoppingCallback, monitor='root_mean_squared_error', min_delta=0.001, patience=10)
            autolstm_learner = Learner(fc_seq_db,autolstm,callback_fns= [partial(OneCycleScheduler,lr_max=lr_max)],loss_func=torch.nn.MSELoss(),metrics=root_mean_squared_error)


            autolstm_learner.fit(max_epoch,lr = lr)
            if plot:
                autolstm_learner.recorder.plot_losses()
                plt.savefig("learning_curve.png")


            
            y_pred,y_true = autolstm_learner.get_preds(DatasetType.Train)
            y_pred = dev_to_np(y_pred.flatten())
            y_true = dev_to_np(y_true.flatten())
            score_train = np.sqrt(np.mean(np.power(y_true-y_pred,2)))


            y_pred,y_true = autolstm_learner.get_preds(DatasetType.Valid)
            y_pred = dev_to_np(y_pred.flatten())
            y_true = dev_to_np(y_true.flatten())
            score_validation = np.sqrt(np.mean(np.power(y_true-y_pred,2)))
            print(f"Forecast RMSE: {np.sqrt(np.mean(np.power(y_true-y_pred,2)))}")

            if score_validation< best_score['validation']:
                best_score = {"validation": score_validation, "train":  score_train}
                best_model = autolstm
                best_params = params

        autolstm_final_eval = Learner(fc_seq_db,best_model,callback_fns= [partial(OneCycleScheduler,lr_max=lr_max)],loss_func=torch.nn.MSELoss(),metrics=root_mean_squared_error)
        
        y_pred,y_true = autolstm_final_eval.get_preds(DatasetType.Test)
        y_pred = dev_to_np(y_pred.flatten())
        y_true = dev_to_np(y_true.flatten())
        score_test = np.sqrt(np.mean(np.power(y_true-y_pred,2)))

        
        results=results.append({'input_data': "reduced" if reduced else "full" ,
                                        'target':cur_name, 'method': "autolstm", 
                                        'best_params': best_params, 'train_score':best_score['train'], 
                                        'train_score_dynamic':best_score['train'],'validation_score':best_score['validation'], 
                                        'validation_socre_dynamic':best_score['validation'], 
                                        "test_score": score_test,
                                        'forecast_horizon':24}, ignore_index=True)

        if reduced:
            torch.save(best_model, f"models/autolstm_reduced_{cur_name}.model")
        else: 
            torch.save(best_model, f"models/autolstm_{cur_name}.model")

        
        results.to_pickle(f"stored_results_autolstm_{'reduced' if reduced else 'full' }.pkl")  
        # # create plots
    # if plot:
        

    #     plt.figure()
    #     plt.plot(y_true, label = "true")
    #     plt.plot(y_pred,alpha=0.5, label="predicted")
    #     plt.ylim(0,1)
    #     plt.legend()
    #     plt.savefig("true_vs_predicted.png")


    #     plt.figure()
    #     plt.plot(y_true[0:72], label = "true")
    #     plt.plot(y_pred[0:72],alpha=0.5, label="predicted")
    #     plt.ylim(0,1)
    #     plt.legend()
    #     plt.savefig("true_vs_predicted_72h.png")



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-j", "--jobs", dest="num_jobs", type="int",default=1)
    parser.add_option("-l", "--lr", dest="lr", type="float",default=0.001)
    parser.add_option("--lrmax", dest="lr_max", type="float",default=0.01)
    parser.add_option("-b", "--batch-size", dest="batch_size", type="int",default=10)
    parser.add_option("-e", "--epochs", dest="epochs", type="int",default=2000)

    parser.add_option("--shuffle-data", dest="shuffle_data",  action="store_true",default=False)
    parser.add_option("--drop_last", dest="drop_last", action="store_true",default=False)
    parser.add_option("-r","--reduced", dest="reduced", action="store_true",default=False)
    parser.add_option("--residual", dest="residual", action="store_true",default=False)
    parser.add_option("--loadae", dest="load_ae_model", action="store_true", default=False)
    parser.add_option("--skip1000s", dest="skip_residual_distributed", action="store_true", default=False)
    parser.add_option("--skip-trained", dest="skip_trained", action="store_true", default=False)
    parser.add_option("--randomized-split-data", dest="random_val_train_test", action="store_true", default=False)
    

    (options, args) = parser.parse_args()
    print(options)
    

    experiment(options)
    

    