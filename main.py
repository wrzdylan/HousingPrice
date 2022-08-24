#!/usr/bin/env python

from GetDataFrame import GetDataFrame


train = GetDataFrame("train")
train.get_cleaned_df()
cleaned_train = train.get_df()

test = GetDataFrame("test")
test.get_cleaned_df()
cleaned_test = test.get_df()

cleaned_train.to_feather('./dataset/cleaned_train.feather')
cleaned_test.to_feather('./dataset/cleaned_test.feather')
