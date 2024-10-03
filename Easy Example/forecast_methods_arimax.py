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
                prediction = fitted_result.get_prediction(dynamic = start)
                end = min(start+self.forecast_horizon, len(y_for_prediction))
                predicted_values = prediction.predicted_mean[start:end]
                dynamic_results.append(predicted_values)
                real_values = y_for_prediction[start:end]
                running_rmse.append(RMSE(real_values,predicted_values))
            self.score_dynamic = np.mean(running_rmse)
            
            # Pad shorter predictions to ensure consistent shape
            max_len = max(len(pred) for pred in dynamic_results)
            padded_results = [np.pad(pred, (0, max_len - len(pred)), 'constant', constant_values=np.nan) for pred in dynamic_results]
            self.results_dynamic = np.array(padded_results)
            
            return np.concatenate(dynamic_results)

        return self.result_static



import converter
   
exemplary_data =  pd.read_csv("exemplary_load.csv", sep=";", decimal=".",thousands=",", 
                             na_values=["-"], dtype={"Datum":str, "Uhrzeit":str}) 
exemplary_data = exemplary_data.iloc[:1000]
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



if __name__ == "__main__":
    arimax = SklearnSarimax(
        dynamic=True,
        use_exogenous_variables=True,
        enforce_invertibility=False,
        enforce_stationary=False,
        forecast_horizon=24,
        initialization="approximate_diffuse",
        return_neg_score=False
    )

    # Ensure input_train and target_train are 2D arrays
    X_train = input_train.values
    y_train = target_train.values.ravel()

    # Check shapes and print them for debugging
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Fit the model
    try:
        arimax.fit(X_train, y_train)
        print("MODEL FITTED SUCCESSFULLY")

        # Calculate scores
        score_dyn_train = arimax.score(input_train, target_train, dynamic=True)
        score_dyn_test = arimax.score(input_test, target_test, dynamic=True)

        print(f"Dynamic Train Score: {score_dyn_train}")
        print(f"Dynamic Test Score: {score_dyn_test}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check the shapes of your input data and target variable.")
    print("MODEL FITTED SUCCESSFULLY")
    print(input_train)
    print()
    print(target_train.values.flatten().shape)
    score_dyn_train=arimax.score(input_train, target_train.values.flatten(), dynamic=True)
    score_dyn_test=arimax.score(input_test, target_test, dynamic=True)

    print(score_dyn_train)



