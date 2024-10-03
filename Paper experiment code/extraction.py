import datetime
import calendar


import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y,check_array
from .ffal.OrthogonalPolynomialSlidingWindow import OrthogonalPolynomialSlidingWindow


class SlidingWindowFeatureExtractor(TransformerMixin, BaseEstimator):


    def __init__(self,
                 window_size = 24,
                 step_size = 1,
                 use_padding=False,
                 padding_value=0
                 ):
        self.window_size = window_size
        self.step_size = step_size

        self.methods = [
            np.mean,
            np.std,
            np.var,
            np.min,
            np.max,
            np.median
        ]

    
    def fit(self, X,y = None):
        return self


    def transform(self,X,y=None):
        X = check_array(X)
        # * [ ] Mean, Var, Std, Min, Max, Median
        len_X = X.shape[0]
        number_of_features = X.shape[1]
        extracted_features = np.zeros((len_X-self.window_size,number_of_features*len(self.methods)))
        for i,(start,end) in enumerate(zip(np.arange(0,len_X-self.window_size,self.step_size), np.arange(self.window_size,len_X,self.step_size))):
            ix = np.arange(start,end)
            cur_data = X[ix,:]
            for j in range(number_of_features):
                extracted_features[i,(j*len(self.methods)):((j+1)*len(self.methods))] = [m(cur_data[:,j]) for m in self.methods]
        return extracted_features

class FastFeatureExtractor(TransformerMixin, BaseEstimator):


    def __init__(self,
                 window_size = 24,
                 step_size = 1,
                 polynom_degree = 4
                 ):
        self.sw = OrthogonalPolynomialSlidingWindow(degree=polynom_degree,window_size=24)
        self.polynom_degree = polynom_degree
        self.window_size = window_size
        self.step_size = step_size

    
    def fit(self, X,y = None):
        return self


    def transform(self,X,y=None):
        X = check_array(X)
        # * [ ] Mean, Var, Std, Min, Max, Median
        len_X = X.shape[0]
        number_of_features = X.shape[1]
        extracted_features = np.zeros((len_X-self.window_size,number_of_features*(self.polynom_degree+1)))
        for i,(start,end) in enumerate(zip(np.arange(0,len_X-self.window_size,self.step_size), np.arange(self.window_size,len_X,self.step_size))):
            ix = np.arange(start,end)
            for j in range(number_of_features):
                for index in ix:
                    self.sw.update(X[index,j])
                extracted_features[i,(j*(self.polynom_degree+1)):((j+1)*(self.polynom_degree+1))] = self.sw.get_orthogonal_coefficients()
                self.sw.restart()
        return extracted_features
 

class TimeShifter(TransformerMixin, BaseEstimator):

    def __init__(self, shift_index_by=1):

        self.shift_index_by = shift_index_by

    def fit(self, X,y=None):
        return self

    def transform(self, X,y=None):
        X = check_array(X)
        return X[(self.shift_index_by):,:]


class TimeFeatureCreator(TransformerMixin, BaseEstimator):

    def __init__(self,
                 year=True,
                 month_of_year = True,
                 week_of_year = True,
                 day_of_year = True,
                 day_of_month = True,
                 day_of_week = True,
                 hour_of_day = True,
                 minute_of_hour = True,
                 do_sin_cos_encoding =True
                 ):
        self.year = year
        self.month_of_year = month_of_year
        self.week_of_year = week_of_year
        self.day_of_year = day_of_year
        self.day_of_month = day_of_month
        self.day_of_week = day_of_week
        self.hour_of_day = hour_of_day
        self.minute_of_hour = minute_of_hour
        self.do_sin_cos_encoding = do_sin_cos_encoding

        self.combined_information = [self.year,
                                     self.month_of_year,
                                     self.week_of_year,
                                     self.day_of_year,
                                     self.day_of_month,
                                     self.day_of_week,
                                     self.hour_of_day,
                                     self.minute_of_hour]

    def __sin_cos_enc(self,data,max_data_possible):
        sin = np.sin(2*np.pi*data/max_data_possible)
        cos = np.cos(2*np.pi*data/max_data_possible)
        return sin,cos

    def fit(self, X,y):
        return self

    def transform(self,X, y=None):
        # TODO: test if X is acutal datetime

        # if using sine cosine encoding add two times the extracted feature size 
        # but non for the year as it can't be ecnoded 
        extracted_feature_size = np.sum(self.combined_information)
        extracted_feature_size = extracted_feature_size +  ((2*extracted_feature_size) if self.do_sin_cos_encoding else 0) - (2 if self.year else 0)
        converted_dates = np.zeros((X.shape[0],extracted_feature_size))

        for ix,dt in enumerate(X):
            i = 0
            tt = dt.timetuple()
            if self.year:
                converted_dates[ix,i] = tt.tm_year
                i = i+1
            if self.month_of_year:
                converted_dates[ix,i] = tt.tm_mon
                i = i + 1
                if (self.do_sin_cos_encoding):
                    converted_dates[ix,i],converted_dates[ix,i+1] = self.__sin_cos_enc(converted_dates[ix,i-1],12)
                    i = i+ 2
                
            if self.week_of_year:
                converted_dates[ix,i] = int(dt.strftime("%W"))
                i = i + 1
                if (self.do_sin_cos_encoding):
                    converted_dates[ix,i],converted_dates[ix,i+1] = self.__sin_cos_enc(converted_dates[ix,i-1],52)
                    i = i+ 2
            if self.day_of_year:
                converted_dates[ix,i] = tt.tm_yday
                i = i + 1
                if (self.do_sin_cos_encoding):
                    converted_dates[ix,i],converted_dates[ix,i+1] = self.__sin_cos_enc(converted_dates[ix,i-1],366)
                    i = i+ 2
            if self.day_of_month:
                converted_dates[ix,i] = tt.tm_mday
                i = i + 1
                if (self.do_sin_cos_encoding):
                    converted_dates[ix,i],converted_dates[ix,i+1] = self.__sin_cos_enc(converted_dates[ix,i-1],31)
                    i = i+ 2
            if self.day_of_week:
                converted_dates[ix,i] = tt.tm_wday
                i = i + 1
                if (self.do_sin_cos_encoding):
                    converted_dates[ix,i],converted_dates[ix,i+1] = self.__sin_cos_enc(converted_dates[ix,i-1],7)
                    i = i+ 2
            if self.hour_of_day:
                converted_dates[ix,i] = tt.tm_hour
                i = i + 1
                if (self.do_sin_cos_encoding):
                    converted_dates[ix,i],converted_dates[ix,i+1] = self.__sin_cos_enc(converted_dates[ix,i-1],24)
                    i = i+ 2
            if self.minute_of_hour:
                converted_dates[ix,i] = tt.tm_min
                i = i + 1
                if (self.do_sin_cos_encoding):
                    converted_dates[ix,i],converted_dates[ix,i+1] = self.__sin_cos_enc(converted_dates[ix,i-1],60)
                    i = i+ 2
        
        return converted_dates


