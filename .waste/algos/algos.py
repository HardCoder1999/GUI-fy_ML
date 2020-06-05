# Algoritms for machine learning classification

# importing statements
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder



class prepare_data:

	# def __init__():

	# Data Preparation
	def prepare_Dataframe(self, filepath):
		dataFrame = pd.read_csv(filepath)
		n_of_rows, n_of_cols = dataFrame.shape
		list_of_col = dataFrame.columns.tolist()
		return dataFrame, n_of_cols, n_of_rows, list_of_col


	def return_feature_label(self, dataFrame, label_column):
		X = dataFrame.drop(labels=[label_column], axis=1)
		y = dataFrame[label_column]
	
		label = LabelEncoder()
		y = label.fit_transform(y)
		y = pd.DataFrame(y)

		return X, y

	def split_data(self, dataFrame, test_size, label_column, filepath):
		X, y = self.return_feature_label(dataFrame, label_column)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=True, shuffle=True)
		self.save_dataframe(X_train, os.path.join(filepath, 'X_train.pkl'))
		self.save_dataframe(X_test, os.path.join(filepath, 'X_test.pkl'))
		self.save_dataframe(y_train, os.path.join(filepath, 'y_train.pkl'))
		self.save_dataframe(y_test, os.path.join(filepath, 'y_test.pkl'))
		data_dict = {
			'X_train' : X_train,
			'X_test' : X_test,
			'y_train': y_train,
			'y_test': y_test
		}
		return data_dict

	def save_dataframe(self, dataFrame, filename):
		dataFrame.to_pickle(filename)

	def load_dataframe(self, filename):
		dataFrame = pd.read_pickle(filename)
		return dataFrame








class knn_model(prepare_data):

	def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2):
		self.n_neighbors = n_neighbors
		self.weights = weights
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.power_parameter = p


	# Model formation and Training
	def train_model(self, X_train, y_train, modelpath):
		model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, 
			algorithm=self.algorithm, p=self.power_parameter)

		model.fit(X_train, y_train)
		self.save_model(model, modelpath)
		return model


	def save_model(self, model, modelpath):
		joblib.dump(model, os.path.join(modelpath, 'knn_model.pkl'))


	# def load_model(self):
	# 	model = joblib.load()
	# 	return model

	def model_score(self, modelpath, X_, y_):
		model = joblib.load(modelpath)
		accuracy_score = model.score(X_, y_)
		return accuracy_score

	def predict_output(self, modelpath, filepath, X_):
		model = joblib.load(modelpath)
		y_pred = model.predict(X_)
	
		y_predicted = pd.DataFrame(y_pred)
		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
		return y_pred








class radius_nn_model(prepare_data):

	def __init__(self, radius=1.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, outlier_label=None):
		self.radius = radius
		self.weights = weights
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.power_parameter = p
		self.outlier_label = outlier_label


	# Model formation and Training
	def train_model(self, X_train, y_train, modelpath):
		model = RadiusNeighborsClassifier(radius=self.radius, weights=self.weights, 
			algorithm=self.algorithm, p=self.power_parameter, outiler_label=self.outlier_label)

		model.fit(X_train, y_train)
		self.save_model(model, modelpath)
		return model


	def save_model(self, model, modelpath):
		joblib.dump(model, os.path.join(modelpath, 'radius_model.pkl'))



	def model_score(self, modelpath, X_, y_):
		model = joblib.load(modelpath)
		accuracy_score = model.score(X_, y_)
		return accuracy_score

	def predict_output(self, modelpath, filepath, X_):
		model = joblib.load(modelpath)
		y_pred = model.predict(X_)
	
		y_predicted = pd.DataFrame(y_pred)
		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
		return y_pred


	def rad_neighbors(self, X_=None, radius=None, return_distance=True, sort_results=False):
		model = joblib.load(modelpath)
		rng = model.radius_neighbors(X_, radius=radius, return_distance=return_distance, 
			sort_results=sort_results)

		return rng



	def rad_neighbors_graph(self, X_=None, radius=None, mode='connectivity', sort_results=False):
		model = joblib.load(modelpath)
		rng = model.radius_neighbors_graph(X=None, radius=None, mode='connectivity', sort_results=False)
		rng = rng.toarray()
		return rng









class svm_model(prepare_data):
	def __init__(self,  C=1.0, kernel='rbf', degree=3, gamma='scale', 
		decision_function_shape='ovr'):
		self.C = C
		self.kernel = kernel
		self.degree = degree
		self.gamma = gamma
		self.decision_function_shape = decision_function_shape



	# Model formation and Training
	def train_model(self, X_train, y_train, modelpath):
		model = SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, 
			decision_function_shape=self.decision_function_shape)

		model.fit(X_train, y_train)
		self.save_model(model, modelpath)
		return model


	def save_model(self, model, modelpath):
		joblib.dump(model, os.path.join(modelpath, 'svc_model.pkl'))



	def model_score(self, modelpath, X_, y_):
		model = joblib.load(modelpath)
		accuracy_score = model.score(X_, y_)
		return accuracy_score

	def predict_output(self, modelpath, filepath, X_):
		model = joblib.load(modelpath)
		y_pred = model.predict(X_)
	
		y_predicted = pd.DataFrame(y_pred)
		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
		return y_pred



	# def decision_func():
	# 	return 1





class linear_svc_model(prepare_data):

	def __init__(penalty='l2', loss='squared_hinge', *, dual=True, C=1.0, multi_class='ovr'):
		self.penalty = penalty
		self.loss = loss
		self.dual = dual
		self.C = C
		self.multi_class = multi_class



	# Model formation and Training
	def train_model(self, X_train, y_train, modelpath):
		model = LinearSVC(penalty=self.penalty, loss=self.loss, dual=self.dual, C=self.C, 
			multi_class=self.multi_class)

		model.fit(X_train, y_train)
		self.save_model(model, modelpath)
		return model


	def save_model(self, model, modelpath):
		joblib.dump(model, os.path.join(modelpath, 'lin_svc_model.pkl'))



	def model_score(self, modelpath, X_, y_):
		model = joblib.load(modelpath)
		accuracy_score = model.score(X_, y_)
		return accuracy_score

	def predict_output(self, modelpath, filepath, X_):
		model = joblib.load(modelpath)
		y_pred = model.predict(X_)
	
		y_predicted = pd.DataFrame(y_pred)
		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
		return y_pred








class decision_tree_model(prepare_data):

	def __init__(self, criterion='gini', splitter='best', max_depth=None, max_features=None):
		self.criterion = criterion
		self.splitter = splitter
		self.max_depth = max_depth
		self.max_features = max_features


	def train_model(self, X_train, y_train, modelpath):
		model = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter, 
			max_depth=self.max_depth, max_features=self.max_features)

		model.fit(X_train, y_train)
		self.save_model(model, modelpath)
		return model


	def save_model(self, model, modelpath):
		joblib.dump(model, os.path.join(modelpath, 'dec_tree_model.pkl'))



	def model_score(self, modelpath, X_, y_):
		model = joblib.load(modelpath)
		accuracy_score = model.score(X_, y_)
		return accuracy_score

	def predict_output(self, modelpath, filepath, X_):
		model = joblib.load(modelpath)
		y_pred = model.predict(X_)
	
		y_predicted = pd.DataFrame(y_pred)
		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
		return y_pred









class logreg_model(prepare_data):

	def __init__(self, penalty='l2', dual=False, C=1.0, fit_intercept=True, 
		solver='lbfgs', multi_class='auto'):
		self.penalty = penalty
		self.dual = dual
		self.C = C
		self.fit_intercept = fit_intercept
		self.solver = solver
		self.multi_class = multi_class



	def train_model(self, X_train, y_train, modelpath):
		model = LogisticRegression(penalty=self.penalty, dual=self.dual, C=self.C, 
			fit_intercept=self.fit_intercept, solver=self.solver, multi_class=self.multi_class)

		model.fit(X_train, y_train)
		self.save_model(model, modelpath)
		return model


	def save_model(self, model, modelpath):
		joblib.dump(model, os.path.join(modelpath, 'log_reg_model.pkl'))



	def model_score(self, modelpath, X_, y_):
		model = joblib.load(modelpath)
		accuracy_score = model.score(X_, y_)
		return accuracy_score

	def predict_output(self, modelpath, filepath, X_):
		model = joblib.load(modelpath)
		y_pred = model.predict(X_)
	
		y_predicted = pd.DataFrame(y_pred)
		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
		return y_pred








class linreg_model(prepare_data):

	def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.copy_X = copy_X
		self.n_jobs = n_jobs



	# Model formation and Training
	def train_model(self, X_train, y_train, modelpath):
		model = LinearRegression(fit_intercept=self.fit_intercept, normalize=self.normalize, 
			copy_X=self.copy_X, n_jobs=self.n_jobs)

		model.fit(X_train, y_train)
		self.save_model(model, modelpath)
		return model


	def save_model(self, model, modelpath):
		joblib.dump(model, os.path.join(modelpath, 'lin_reg_model.pkl'))



	def model_score(self, modelpath, X_, y_):
		model = joblib.load(modelpath)
		accuracy_score = model.score(X_, y_)
		return accuracy_score

	def predict_output(self, modelpath, filepath, X_):
		model = joblib.load(modelpath)
		y_pred = model.predict(X_)
	
		y_predicted = pd.DataFrame(y_pred)
		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
		return y_pred




class gaussian_nb_model(prepare_data):

	# Model formation and Training
	def train_model(self, X_train, y_train, modelpath):
		model = GaussianNB()

		model.fit(X_train, y_train)
		self.save_model(model, modelpath)
		return model


	def save_model(self, model, modelpath):
		joblib.dump(model, os.path.join(modelpath, 'gauss_nb_model.pkl'))



	def model_score(self, modelpath, X_, y_):
		model = joblib.load(modelpath)
		accuracy_score = model.score(X_, y_)
		return accuracy_score

	def predict_output(self, modelpath, filepath, X_):
		model = joblib.load(modelpath)
		y_pred = model.predict(X_)
	
		y_predicted = pd.DataFrame(y_pred)
		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
		return y_pred

	




class multinomial_nb_model(prepare_data):

	# Model formation and Training
	def train_model(self, X_train, y_train, modelpath):
		model = MultinomialNB()

		model.fit(X_train, y_train)
		self.save_model(model, modelpath)
		return model


	def save_model(self, model, modelpath):
		joblib.dump(model, os.path.join(modelpath, 'multi_nb_model.pkl'))



	def model_score(self, modelpath, X_, y_):
		model = joblib.load(modelpath)
		accuracy_score = model.score(X_, y_)
		return accuracy_score

	def predict_output(self, modelpath, filepath, X_):
		model = joblib.load(modelpath)
		y_pred = model.predict(X_)
	
		y_predicted = pd.DataFrame(y_pred)
		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
		return y_pred

	



class bernoulli_nb_model(prepare_data):

	# Model formation and Training
	def train_model(self, X_train, y_train, modelpath):
		model = BernoulliNB()

		model.fit(X_train, y_train)
		self.save_model(model, modelpath)
		return model


	def save_model(self, model, modelpath):
		joblib.dump(model, os.path.join(modelpath, 'bern_nb_model.pkl'))



	def model_score(self, modelpath, X_, y_):
		model = joblib.load(modelpath)
		accuracy_score = model.score(X_, y_)
		return accuracy_score

	def predict_output(self, modelpath, filepath, X_):
		model = joblib.load(modelpath)
		y_pred = model.predict(X_)
	
		y_predicted = pd.DataFrame(y_pred)
		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
		return y_pred

	



class random_forest_model(prepare_data):

	def __init__(self, n_estimators=100, criterion='gini'):
		self.n_estimators = n_estimators
		self.criterion = criterion




	# Model formation and Training
	def train_model(self, X_train, y_train, modelpath):
		model = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion)

		model.fit(X_train, y_train)
		self.save_model(model, modelpath)
		return model


	def save_model(self, model, modelpath):
		joblib.dump(model, os.path.join(modelpath, 'rf_model.pkl'))



	def model_score(self, modelpath, X_, y_):
		model = joblib.load(modelpath)
		accuracy_score = model.score(X_, y_)
		return accuracy_score

	def predict_output(self, modelpath, filepath, X_):
		model = joblib.load(modelpath)
		y_pred = model.predict(X_)
	
		y_predicted = pd.DataFrame(y_pred)
		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
		return y_pred








# class bagging_clf_model(prepare_data):

# 	def __init__():






# 	# Model formation and Training
# 	def train_model(self, X_train, y_train, modelpath):
# 		model = BaggingClassifier()

# 		model.fit(X_train, y_train)
# 		self.save_model(model, modelpath)
# 		return model


# 	def save_model(self, model, modelpath):
# 		joblib.dump(model, os.path.join(modelpath, 'bern_nb_model.pkl'))



# 	def model_score(self, modelpath, X_, y_):
# 		model = joblib.load(modelpath)
# 		accuracy_score = model.score(X_, y_)
# 		return accuracy_score

# 	def predict_output(self, modelpath, filepath, X_):
# 		model = joblib.load(modelpath)
# 		y_pred = model.predict(X_)
	
# 		y_predicted = pd.DataFrame(y_pred)
# 		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
# 		return y_pred








# class adaboost_clf_model(prepare_data):

# 	def __init__():







# 	# Model formation and Training
# 	def train_model(self, X_train, y_train, modelpath):
# 		model = AdaBoostClassifier()

# 		model.fit(X_train, y_train)
# 		self.save_model(model, modelpath)
# 		return model


# 	def save_model(self, model, modelpath):
# 		joblib.dump(model, os.path.join(modelpath, 'bern_nb_model.pkl'))



# 	def model_score(self, modelpath, X_, y_):
# 		model = joblib.load(modelpath)
# 		accuracy_score = model.score(X_, y_)
# 		return accuracy_score

# 	def predict_output(self, modelpath, filepath, X_):
# 		model = joblib.load(modelpath)
# 		y_pred = model.predict(X_)
	
# 		y_predicted = pd.DataFrame(y_pred)
# 		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
# 		return y_pred







# class gradboost_clf_model(prepare_data):

# 	def __init__():









# 	# Model formation and Training
# 	def train_model(self, X_train, y_train, modelpath):
# 		model = GradientBoostingClassifier()

# 		model.fit(X_train, y_train)
# 		self.save_model(model, modelpath)
# 		return model


# 	def save_model(self, model, modelpath):
# 		joblib.dump(model, os.path.join(modelpath, 'bern_nb_model.pkl'))



# 	def model_score(self, modelpath, X_, y_):
# 		model = joblib.load(modelpath)
# 		accuracy_score = model.score(X_, y_)
# 		return accuracy_score

# 	def predict_output(self, modelpath, filepath, X_):
# 		model = joblib.load(modelpath)
# 		y_pred = model.predict(X_)
	
# 		y_predicted = pd.DataFrame(y_pred)
# 		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
# 		return y_pred








# class voting_clf_model(prepare_data):

# 	def __init__():









# 	# Model formation and Training
# 	def train_model(self, X_train, y_train, modelpath):
# 		model = VotingClassifier()

# 		model.fit(X_train, y_train)
# 		self.save_model(model, modelpath)
# 		return model


# 	def save_model(self, model, modelpath):
# 		joblib.dump(model, os.path.join(modelpath, 'bern_nb_model.pkl'))



# 	def model_score(self, modelpath, X_, y_):
# 		model = joblib.load(modelpath)
# 		accuracy_score = model.score(X_, y_)
# 		return accuracy_score

# 	def predict_output(self, modelpath, filepath, X_):
# 		model = joblib.load(modelpath)
# 		y_pred = model.predict(X_)
	
# 		y_predicted = pd.DataFrame(y_pred)
# 		self.save_dataframe(y_predicted, os.path.join(filepath, 'y_pred.pkl'))
# 		return y_pred








