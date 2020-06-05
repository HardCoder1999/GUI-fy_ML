import pandas as pd
import numpy as np

def panda(file_path):
	df = pd.read_csv(file_path)
	tables=[df.to_html(classes='data')]
	titles=df.columns.values

	return [tables, titles]
