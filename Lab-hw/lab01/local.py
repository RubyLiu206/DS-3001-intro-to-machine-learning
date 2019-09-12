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
print("result nan:", result_missing_values)

new_dataframe = getDfSummary(ads)
result_binary = new_dataframe[new_dataframe.Number_Distinct<=2]
print(" result binary:",result_binary)





# Place your code here
# Place your code here
# seeing the dataset we can find that the Nan values always happen in [visit_freq]
# try to contain the data only with Nan
#Then use the getDfSummary to analysis the new dataframe

dataframe_only_with_nan = ads[ads["buy_freq"].isnull()]
new_dataframe = getDfSummary(dataframe_only_with_nan)
old_dataframe = getDfSummary(ads)
print(new_dataframe)
print(old_dataframe)
# according to the staff we did before, we can conclude that [isbuyer] always be 0 or 1, and to look at the daraframe, when [isbuyer] = 0, the [visit_freq] = Nan
# so the first thing want to try is to focus on the correlation between [isbuyer] and [buy_freq]
# the function I choose is corr() from pandas
correlation_between_isbuyer_buy_freq = dataframe_only_with_nan.isbuyer.corr(dataframe_only_with_nan['buy_freq'])
print(correlation_between_isbuyer_buy_freq)
# but the result show Nan, so what we can do next step is try to figure out all the correlations between each variances using the corr function
print(ads.buy_freq.corr(ads['buy_interval']))
print(ads.buy_interval.corr(ads['expected_time_buy']))
# Conclusion :
# So we can only look at the describle charts: [buy_interval] this feature from 1 to 285 in number distinct, and [expected_time_buy] also change from 1 to 348
# if [buy_interval] = 0 and [expected_time_buy] = 0, then will high probability the [buy_freq] = Nan
# also if youonly look at the chart, we can find that, when [buy_freq]>1, [multiple_buy] is 1. Combining with the former result we have about [isbuyer], when [isbuyer] = 0, [multiple_buy] = 0, we can infor [buy_freq] = nan


