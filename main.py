#!/usr/bin/env python

from GetDataFrame import GetDataFrame


train = GetDataFrame("train")
cleaned_train = train.get_cleaned_df()
# print(cleaned_train.head())
# print(cleaned_train.info())

cleaned_train.to_feather('./dataset/cleaned_train.feather')
