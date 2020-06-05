import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer



# def return_dataframe(filepath):
# 	dataFrame = pd.read_csv(filepath)
# 	return dataFrame

def data_prep(dataFrame):

	# row and column value of dataFrame
	row_val = dataFrame.shape[0]
	col_val = dataFrame.shape[1]

	list_of_columns = dataFrame.columns

	column_description = dataFrame[column].describe()

	total_columns = len(list_of_columns)

	return (row_val, col_val, list_of_columns, column_description)



def column_description(dataFrame, column):
	# returning the description of the column of dataFrame
	column_description_values = dataFrame[column].describe()
	columns_description_keys = column_description_values.index
	return column_description_keys, column_description_values


def missing_values(dataFrame):
	mis_col = list(dataFrame.columns[data.isnull().any()])
	mis_col_val = list(dataFrame[z].isnull().sum())
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
	#datafFrame.apply(label.fit_transform)
	label = LabelEncoder()
	for i in list_of_columns:
		dataFrame[i] = label.fit_transform(dataFrame[i])
	return dataFrame

def normalization(dataFrame, string_val):
	
	min_max_scaler = MinMaxScaler()
	robust_scaler = RobustScaler()
	standard_scaler = StandardScaler()
	normalizer = Normalizer()

	if string_val == 'MinMaxScaler':
		dataFrame = min_max_scaler.fit_transform(dataFrame)

	elif string_val == 'RobustScaler':
		dataFrame = robust_scaler.fit_transform(dataFrame)

	elif string_val == 'StandardScaler':
		dataFrame = standard_scaler.fit_transform(dataFrame)

	elif string_val == 'Normalizer':
		dataFrame = normalizer.fit_transform(dataFrame)

	return dataFrame

	
	def save_dataframe(dataFrame, filename):
		dataFrame.to_pickle(filename)

	def load_dataframe(filename):
		dataFrame = pd.read_pickle(filename)
		return dataFrame

