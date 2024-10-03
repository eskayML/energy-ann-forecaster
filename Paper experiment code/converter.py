import datetime
import calendar


import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y,check_array

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
        self.__feature_names = ["year",
                                "month_of_year",
                                "week_of_year",
                                "day_of_year",
                                "day_of_month",
                                "day_of_week",
                                "hour_of_day",
                                "minute_of_hour"]

    def __sin_cos_enc(self,data,max_data_possible):
        sin = np.sin(2*np.pi*data/max_data_possible)
        cos = np.cos(2*np.pi*data/max_data_possible)
        return sin,cos

    def fit(self, X,y):
        return self

    def transform(self,X, y=None):
        # TODO: test if X is acutal datetime
        if len(X)>0 and (type(X) == pd.DataFrame or type(X) == pd.Series):
            X = X.index
        

        # if using sine cosine encoding add two times the extracted feature size 
        # but non for the year as it can't be ecnoded 
        extracted_feature_size = np.sum(self.combined_information)
        extracted_feature_size = extracted_feature_size +  ((2*extracted_feature_size) if self.do_sin_cos_encoding else 0) - (2 if self.year else 0)
        converted_dates = np.zeros((X.shape[0],extracted_feature_size))
        
        self.extracted_features_name = []
        
        for i,name in enumerate(self.__feature_names):
            if (self.combined_information[i]):
                self.extracted_features_name.extend([name])
                if (self.do_sin_cos_encoding and name != "year"):
                    self.extracted_features_name.extend([f"{name}_{method}" for method in ["sin","cos"]])
            

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
