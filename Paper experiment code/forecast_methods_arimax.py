import statsmodels.api as sm
import pandas as pd 
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y

class SklearnSarimax(BaseEstimator, RegressorMixin):

    def __init__(self, 
                 order=(1, 0, 0), 
                 seasonal_order=(0, 0, 0, 0), 
                 dynamic=False, 
                 use_exogenous_variables=True, 
                 enforce_stationary=False, 
                 enforce_invertibility=False, 
                 initialization='approximate_diffuse',
                 forecast_horizon=24,
                 return_neg_score=False):
        self.order = order
        self.seasonal_order=seasonal_order
        self.enforce_stationary = enforce_stationary
        self.enforce_invertibility = enforce_invertibility
        self.initialization = initialization
        self.forecast_horizon = forecast_horizon
        self.use_exogenous_variables = use_exogenous_variables
        self.dynamic = dynamic
        self.return_neg_score = return_neg_score

    def __create_new_model(self,X, y):
        # print(y)
        return sm.tsa.statespace.SARIMAX(y, 
                                  order=self.order, 
                                  exog= X if self.use_exogenous_variables else None,
                                  seasonal_order=self.seasonal_order, 
                                  enforce_stationarity=self.enforce_stationary,
                                  enforce_invertibility=self.enforce_invertibility,
                                  initialization=self.initialization)
  

    def fit(self, X,y):
        # if (y is not None):
        X,y = check_X_y(X,y)

        self._X_initial_model = X
        self._y_initial_model = y

        self.model = self.__create_new_model(X if self.use_exogenous_variables else None,y)
        
        self.result = self.model.fit(disp=False)

        
        return self
    
    def score(self, X, y, dynamic=False):
        X,y = check_X_y(X,y)
        self.predict(X,y)
        if (self.dynamic or dynamic):
            return self.score_dynamic * (-1 if self.return_neg_score else 1)
        else:
            return self.score_static * (-1 if self.return_neg_score else 1)

    
    def predict(self, X, y=None, dynamic=False):
        # some formal checks
        X = X.copy()
        
        if (y is not None):
            y = y.copy()
            X,y = check_X_y(X,y)
            len_y = y.shape[0]
            
            y_for_prediction = np.hstack((self._y_initial_model,y))
            if (self.use_exogenous_variables):
                X_for_prediction = np.vstack((self._X_initial_model,X))
        else:
            if self.use_exogenous_variables:
                raise ValueError("you cannot use exogeneous variables without giving the endogeneous variable")
                # X_for_prediction = np.hstack((self._X_initial_model,X))
            else:
                y_for_prediction = np.hstack((self._y_initial_model,X))
                len_y = X.shape[0]
        

        RMSE = lambda real, predicted: np.sqrt(np.mean(np.power(real-predicted,2)))
        
        self._fitted_model = self.__create_new_model(X_for_prediction if self.use_exogenous_variables else None, y_for_prediction)
        fitted_result = self._fitted_model.filter(self.result.params)
        
        prediction = fitted_result.get_prediction(dynamic = False)
        self.result_static = prediction.predicted_mean[(len(y_for_prediction)-len_y):]
        real_values = y_for_prediction[(len(y_for_prediction)-len_y):]
        self.score_static = RMSE(real_values, self.result_static)
        
        if (self.dynamic or dynamic):
            running_rmse = list()
            dynamic_results = list()
            steps = np.arange(len(y_for_prediction)-len_y,len(y_for_prediction),self.forecast_horizon)
            for start in steps:
                # int(len(y)/self.forecast_horizon)
                prediction = fitted_result.get_prediction(dynamic = start)
                # start = len(y_for_prediction)+step
                end = start+self.forecast_horizon
                predicted_values = prediction.predicted_mean[start:end]
                dynamic_results.append(predicted_values)
                real_values = y_for_prediction[start:end]
                running_rmse.append(RMSE(real_values,predicted_values))
            self.score_dynamic = np.mean(running_rmse)
            self.results_dynamic = np.array(dynamic_results)
            return np.concatenate(self.results_dynamic)

        return self.result_static
