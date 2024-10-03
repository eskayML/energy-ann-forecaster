import os
import pickle
import pandas as pd
import numpy as np

##wozu braucht man den logger?
import logger as log

from abc import ABCMeta, abstractmethod
from multiprocessing import Pool, TimeoutError
from sklearn.preprocessing import StandardScaler

from itertools import product

def extract_headers(data):
    if type(data) == pd.DataFrame:
        return data.columns
    elif type(data) == pd.Series:
        return [data.name]
    return None

class Dataset(metaclass=ABCMeta):
    __version__ = 2
    __name__ = "maindataset"

    def __init__(self,folder="", splits=(0.7,0.2,0.1),feature_extraction_pipeline=None, target_preprocessor=None, data_scaler=StandardScaler, nr_parallel_file_reader=4, use_cache=True):
        # check splits sum up to 1
        if not np.allclose(np.sum(splits),1):
            raise ValueError(f"splits should sum up to 1 but instead sum up to {'+'.join(map(str,splits))}={splits}")
        self.folder = folder
        self.feature_extraction_pipeline = feature_extraction_pipeline
        self.data_scaler = data_scaler
        self.splits = splits
        self.nr_parallel_file_reader = nr_parallel_file_reader
        self.target_preprocessor = target_preprocessor

        self.original_data = {
            'names': list(),
            'inputs': list(),
            'targets': list(),
            'input_scaler': list(),
            'target_scaler': list()
        }

        self.train_data = {
            'names': list(),
            'inputs': list(),
            'targets': list()
        }
        self.test_data = {
            'names': list(),
            'inputs': list(),
            'targets': list()
        }
        self.validation_data = {
            'names': list(),
            'inputs': list(),
            'targets': list()
        }
        self.use_cache = use_cache

        did_use_cache = False
        
        # check of cache file if available load cache
        log.debug(self.__name__, f"start reading in dataset")
        if (os.path.isfile(f"{self.folder}/cached_data.pickle") and self.use_cache):
            log.info(__name__,"using cache")
            did_use_cache = self.__load_from_cache()
        
        if not did_use_cache:
            log.info(__name__,"not using cache")
            self.read_in_data(self.folder)
            if (self.use_cache):
                log.info(__name__,"writing to cache")
                self.__write_to_cache()
        
        

        super().__init__()
        
    
    @abstractmethod
    def file_filter(self,files):
        pass

    @abstractmethod
    def read_a_file(self,folder,filename):
        pass

    def __write_to_cache(self):
        with open(f"{self.folder}/cached_data.pickle","wb") as file:
            data = {
                "original_data": self.original_data,
                "train_data": self.train_data,
                "test_data": self.test_data,
                "validation_data": self.validation_data,
                "version": self.__version__,
                "feature_extraction_pipeline": self.feature_extraction_pipeline,
                "target_preprocessor": self.target_preprocessor,
                
            }
            pickle.dump(data,file)
            file.close()

    def __load_from_cache(self):
        
        with open(f"{self.folder}/cached_data.pickle","rb") as file:
            data = pickle.load(file)
            if 'version' not in data.keys() or data['version'] != self.__version__:
                log.warn(__name__, "Datafile version mismatch read in data again")
                return False
            self.original_data = data["original_data"]
            self.train_data = data["train_data"]
            self.test_data = data["test_data"]
            self.validation_data = data["validation_data"]
            self.feature_extraction_pipeline = data['feature_extraction_pipeline']
            self.target_preprocessor = data['target_preprocessor']
            return True


    def _read_in_data_file(self, folder, file):
        log.debug(__name__, f"now reading file {folder}/{file}")
        data, target, name = self.read_a_file(folder,file)
        
        
        cur_length_of_data = data.shape[0]

        # self.length_of_data = max(self.original_data['inputs'][0].shape[0],cur_length_of_data)
        train_data = {
            'names': list(),
            'inputs': list(),
            'targets': list()
        }
        validation_data = {
            'names': list(),
            'inputs': list(),
            'targets': list()
        }
        test_data = {
            'names': list(),
            'inputs': list(),
            'targets': list()
        }

        last_split_end = 0
        split_sum = 0
        target_scaler = self.data_scaler() if self.data_scaler is not None else None
        input_scaler = self.data_scaler() if self.data_scaler is not None else None
        for split,data_partition,train_scaler in zip(self.splits,[train_data,validation_data,test_data],[True, False, False]):
            split_sum = split+split_sum
            cur_split_end = int(cur_length_of_data*(split_sum))
            cur_data = data.iloc[last_split_end:cur_split_end]
            cur_target = target.iloc[last_split_end:cur_split_end]
            log.debug(__name__,f"split from {last_split_end} to {cur_split_end}")
            feature_header = extract_headers(cur_data)
            feature_index = cur_data.index

            if self.feature_extraction_pipeline is not None:
                cur_data = self.feature_extraction_pipeline.fit_transform(cur_data)
                if (type(cur_data) == pd.DataFrame or type(cur_data) == pd.Series):
                    feature_header = extract_headers(cur_data)
                    feature_index = cur_data.index
                else:
                    feature_header = None
                    feature_index=None

            target_header = extract_headers(cur_target)
            target_index = cur_target.index

            if self.target_preprocessor is not None:
                cur_target = self.target_preprocessor.fit_transform(cur_target)
                if (type(cur_target) == pd.DataFrame or type(cur_target) == pd.Series):
                    target_header = extract_headers(cur_target)
                    target_index = cur_data.index
                else:
                    cur_target = cur_target.reshape(-1,1)
                    target_header = None
                    target_index=None

            if train_scaler and self.data_scaler is not None:
                cur_data = pd.DataFrame(input_scaler.fit_transform(cur_data),columns = feature_header, index = feature_index)
                cur_target = pd.DataFrame(target_scaler.fit_transform(cur_target),columns = target_header, index = target_index)
            elif not train_scaler and self.data_scaler is not None:
                cur_data = pd.DataFrame(input_scaler.transform(cur_data),columns = feature_header, index = feature_index)
                cur_target = pd.DataFrame(target_scaler.transform(cur_target),columns = target_header, index = target_index)
            
            data_partition['inputs'] = cur_data
            data_partition['targets'] = cur_target
            data_partition['names'] = name
        
            last_split_end = cur_split_end+1
        return data, target, name, train_data, validation_data, test_data, input_scaler, target_scaler

    def read_in_data(self,folder):

        files = os.listdir(folder)
        csv_files = self.file_filter(files)

        with Pool(processes=self.nr_parallel_file_reader) as pool:
            
            parallel_read_in_results = pool.starmap(self._read_in_data_file,product([folder],csv_files))
            for data in parallel_read_in_results:
                self.original_data['inputs'].append(data[0])
                self.original_data['targets'].append(data[1])
                self.original_data['names'].append(data[2])

                for data_preprocessd, data_partition in zip(data[3:6],[self.train_data,self.validation_data,self.test_data]):
                    data_partition['inputs'].append(data_preprocessd['inputs'])
                    data_partition['targets'].append(data_preprocessd['targets'])
                    data_partition['names'].append(data_preprocessd['names'])
                self.original_data['input_scaler'].append(data[6])
                self.original_data['target_scaler'].append(data[7])

# csells/hessian load data

class HessianLoadDataset(Dataset):
    __name__ = "HessianLoadDataset"

    def __init__(self,folder="", splits=(0.7,0.2,0.1),feature_extraction_pipeline=None,target_preprocessor=None, use_cache=True,data_scaler=StandardScaler, nwp_model="gfs", nwp_forecast_horizon=24, nr_parallel_file_reader=4):
        self.nwp_forecast_horizon = nwp_forecast_horizon
        self.nwp_model = nwp_model
        

        if nwp_model == "gfs":
            self.read_in_nwp(f"{folder}/gfs.csv",nwp_forecast_horizon)
        elif nwp_model == "ecmwf":
            self.read_in_nwp(f"{folder}/ecmwf.csv",nwp_forecast_horizon)
        super().__init__(folder=folder,splits=splits,feature_extraction_pipeline=feature_extraction_pipeline,target_preprocessor=target_preprocessor,data_scaler=data_scaler,use_cache=use_cache,nr_parallel_file_reader=nr_parallel_file_reader)


    def read_in_nwp(self, filename, forecast_horizon):
        nwp = pd.read_csv(filename,index_col=1)
        nwp = nwp.drop(nwp.columns[0],axis=1)

        index_df = pd.DataFrame([pd.to_datetime(nwp.index),nwp['forecast_horizon'].values]).transpose()
        index_df.columns = ["time","forecast_horizon"]
        
        nwp.index = pd.MultiIndex.from_frame(index_df)
        nwp = nwp.drop(['forecast_horizon'],axis=1)

        nwp_one_forecast_horizon = nwp.query(f'forecast_horizon=={forecast_horizon}')
        nwp_one_forecast_horizon = nwp_one_forecast_horizon.reset_index('forecast_horizon')
        nwp_one_forecast_horizon = nwp_one_forecast_horizon.drop(['forecast_horizon'],axis=1)
        
        self.nwp = nwp_one_forecast_horizon.resample("1h").interpolate()

    def file_filter(self,files):
        return [f for f in files if ".csv" == f[-4:] and "gfs" not in f and "ecmwf" not in f]
    
    def read_a_file(self,folder,filename):
        data = pd.read_csv(f"{folder}/{filename}",sep=",")
        
        data.index = pd.to_datetime(data['DateTime'])
        data = data.drop(data.columns[[0,1]],axis=1)
        
        # make sure only to use overlapping indices
        longer_indices,shorter_indices = (self.nwp.index, data.index) if self.nwp.index.shape[0] > data.index.shape[0] else (self.nwp.index, data.index)
        ix_to_use = [i for i in shorter_indices if i in longer_indices] 

        target = data.loc[ix_to_use]
        data = self.nwp.loc[ix_to_use]
        return data, target, filename[:-4]


# eex data: https://projecteuclid.org/euclid.aoas/1380804807#supplemental