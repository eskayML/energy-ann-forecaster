import torch
import numpy as np
import pandas as pd

def np_to_dev(xx, device=None, dtype=torch.FloatTensor):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return torch.from_numpy(xx).type(dtype).to(device)

def dev_to_np(xx):
    if type(xx) == tuple: return tuple([dev_to_np(xx_i) for xx_i in xx])
    if type(xx) == list: return [dev_to_np(xx_i) for xx_i in xx]
    return xx.cpu().detach().numpy()



def convert_ts_data_to_cnn_ts_data(data, timesteps=24):
        conv = int(data.shape[0]/timesteps)
        n_features = data.shape[1]
        data_new = np.zeros((conv-1,n_features,timesteps))
        for i,ix in enumerate(range(int(data.shape[0]/timesteps)-1)):
            start = ix*timesteps
            end = (ix+1)*timesteps
            data_new[i,:,:]  = data.swapaxes(0,1).iloc[:,start:end]
        return torch.from_numpy(data_new).type(torch.FloatTensor)
    
    
class FastAICompatibleDataSet():

    def __init__(self,X,y, device="cpu"):
        if (isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)):
            
            self._columns = X.columns
            self._index_x = X.index
        
        self.device = device

        self.X, self.y = self._to_tensor(X), self._to_tensor(y)

        if len(self.y.shape) == 1:
            self.y = np.reshape(self.y, (-1, 1))
    
    def to_device(self, device):
        self.X = self.X.to(device)
        self.y = self.y.to(device)

    def to_np(self):
        return dev_to_np(self.X), dev_to_np(self.y)

    def _to_tensor(self, data, dtype=torch.FloatTensor):
        if (isinstance(data, pd.DataFrame) or isinstance(data, pd.Series)):
            data = data.values
            
            
        if isinstance(data, np.ndarray):
            data = np_to_dev(data, self.device, dtype=dtype)

        return data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]