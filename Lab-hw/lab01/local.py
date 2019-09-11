# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:12:33 2019

@author: ruby_
"""

import numpy as np
import pandas as pd
import timeit


ads = pd.read_csv('ads_dataset.tsv', sep = '\t',encoding='utf-8')
print(ads)


def getDfSummary(input_data):
    # Place your code here
    #new a new data frame
    #for each variable compute the number_nan, num_distinct, mean, max, min, std
    #put the input_data's column to the output_data's index
    #put the new features to the output_data's columns
    each_row = []
    for feature in input_data:
        num_nan = np.count_nonzero(input_data[feature].isnull())
        num_distinct = input_data[feature].nunique()
        des = input_data[feature].describe()[['mean','max','min','std','25%','50%','75%']].values.tolist()
        each_row.append([num_nan,num_distinct]+des)
    output_data = pd.DataFrame(each_row)
    output_data.columns = ['Number_NaN','Number_Distinct','Mean','Max','Min','Std','25%','50%','75%']
    output_data.index = [input_data.columns]
    return output_data


new_dataframe = getDfSummary(ads)
result_missing_values = new_dataframe[new_dataframe.Number_NaN>0]
print(result_missing_values)

new_dataframe = getDfSummary(ads)
result_binary = new_dataframe[new_dataframe.Number_Distinct<=2]
print(result_binary)

# Place your code here
new_dataframe = getDfSummary(ads)
result_missing_values = new_dataframe[new_dataframe.Number_NaN>0]
# seeing the dataset we can find that the Nan values always happen in [visit_freq]
# try to drop that variance using the dropna()
dataframe_after_drop = ads.dropna()
#Then use the getDfSummary to analysis the new dataframe
Summary_after_drop = getDfSummary(dataframe_after_drop)

# according to the staff we did before, we can conclude that [isbuyer] always be 0 or 1, and to look at the daraframe, when [isbuyer] = 0, the [visit_freq] = Nan
# so the first thing want to try is to focus on the correlation between [isbuyer] and [visit_freq]
# the function I choose is corr() from pandas

correlation_between_isbuyer_buy_freq = dataframe_after_drop.isbuyer.corr(dataframe_after_drop['buy_freq'])
print(correlation_between_isbuyer_buy_freq)
# but the result show Nan, so what we can do next step is try to figure out all the correlations between each variances using the corr function

correlation_between_all_columns = dataframe_after_drop.corr()
print(correlation_between_all_columns)
print(correlation_between_all_columns.buy_freq)
# output the columns [buy_freq] show us all the correlations between [buy_freq] with others
# the result is isbuyer                     NaN
#buy_freq               1.000000
#visit_freq             0.487548
#buy_interval           0.398839
#sv_interval            0.000280
#expected_time_buy     -0.291767
#expected_time_visit    0.060817
#last_buy              -0.126793
#last_visit            -0.126793
#multiple_buy           0.735054
#multiple_visit         0.154837
#uniq_urls              0.042624
#num_checkins           0.042764
#y_buy                  0.128118

# the highest one is [multiple_buy]
# then print that two columns, to see the relation

between_buy_freq_multiple = dataframe_after_drop[['buy_freq','multiple_buy']]
print(between_buy_freq_multiple)
# in which we can conclude that when [buy_freq]>1, [multiple_buy] is 1
# combine with the former result we have about [isbuyer], when [isbuyer] = 0, [multiple_buy] = 0, we can infor [buy_freq] = 0



