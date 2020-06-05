import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
import os
import json



# def return_dataframe(filepath):
# 	dataFrame = pd.read_csv(filepath)
# 	return dataFrame

def data_preparation(dataFrame):

	# row and column value of dataFrame
	row_val = dataFrame.shape[0]
	col_val = dataFrame.shape[1]

	list_of_columns = dataFrame.columns

	total_columns = len(list_of_columns)

	return (row_val, col_val, list_of_columns)



def column_description(dataFrame, column):
	# returning the description of the column of dataFrame
	column_description_values = dataFrame[column].describe()
	column_description_keys = column_description_values.index
	return column_description_keys, column_description_values


def missing_values(dataFrame):
	mis_col = list(dataFrame.columns[dataFrame.isnull().any()])
	mis_col_val = list(dataFrame[mis_col].isnull().sum())
	return (mis_col, mis_col_val)


def drop_missing_cols(dataFrame):
	dataFrame = dataFrame.dropna(axis=1)
	return dataFrame

def drop_missing_rows(dataFrame):
	dataFrame = dataFrame.dropna(axis=0)
	return dataFrame


def change_col_name(dataFrame, old_column_name, new_column_name):
	dataFrame = dataFrame.rename(columns={
		old_column_name: new_column_name 
	})
	return dataFrame

def label_encoder(dataFrame, list_of_columns):
	label = LabelEncoder()
	for i in list_of_columns:
		dataFrame[i] = label.fit_transform(dataFrame[i])
	return dataFrame


def standard_(dataFrame):
	scaler = StandardScaler()
	dataFrame = scaler.fit_transform(dataFrame)
	dataFrame = pd.DataFrame(dataFrame)
	return dataFrame


def min_max_(dataFrame):
	scaler = MinMaxScaler()
	dataFrame = scaler.fit_transform(dataFrame)
	dataFrame = pd.DataFrame(dataFrame)
	return dataFrame


def robust_(dataFrame):
	scaler = RobustScaler()
	dataFrame = scaler.fit_transform(dataFrame)
	dataFrame = pd.DataFrame(dataFrame)
	return dataFrame

def normalization_(dataFrame):
	scaler = Normalizer()
	dataFrame = scaler.fit_transform(dataFrame)
	dataFrame = pd.DataFrame(dataFrame)
	return dataFrame

	
def save_dataframe(dataFrame, filename):
	dataFrame.to_pickle(filename)

def load_dataframe(filename):
	dataFrame = pd.read_pickle(filename)
	return dataFrame

def list_all_dataframes(filepath):
	contents = os.listdir(filepath)
	return contents


def drop_cols(dataFrame, col_list):
	dataFrame = dataFrame.drop(columns=col_list)
	return dataFrame