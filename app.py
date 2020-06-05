from flask import *
from werkzeug.utils import secure_filename
import os
from algos.pandas_ import panda
import algos.data_prep as dp
import algos.data_plot as dplot
import pandas as pd
from io import StringIO
import base64
import matplotlib.pyplot as plt
import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from algos.algorithms import *
#import data_prep as dp


UPLOAD_FOLDER = 'static/data'
IMAGE_FOLDER = 'static/data_image'
MODEL_FOLDER = 'static/models'


# creating the application
app = Flask(__name__)


# application configurations
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# setting up secret key for my application
app.secret_key = 'jarvis'



# creating routing functions
@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		datafile = request.files['datafile']
		if datafile:
			filename = secure_filename(datafile.filename)
			path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			datafile.save(path)
			file = pd.read_csv(path)
			dp.save_dataframe(file, os.path.join(app.config['UPLOAD_FOLDER'], 'file.pkl'))
		session['response'] = path
	return redirect(url_for('load_data'))



@app.route('/load-data', methods=['GET', 'POST'])
def load_data():
	row = 0
	col = 0
	list_of_col = []
	col_desc_keys = []
	col_desc_val = []
	length_of_data = 0
	len_mis =0
	mis_col = []
	mis_col_val = []
	path = session['response']
	[tables, titles] = panda(path)
	dataframes_list = dp.list_all_dataframes(app.config['UPLOAD_FOLDER'])

	file = request.args.get('file')
	if file is not None:
		dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))
		(row, col, list_of_col) = dp.data_preparation(dataFrame)
		(mis_col, mis_col_val) = dp.missing_values(dataFrame)
		len_mis = len(mis_col)
		session['file'] = file

	column = request.args.get('column')
	if column is not None:
		#dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))
		(col_desc_keys, col_desc_val) = dp.column_description(dataFrame, column)
		length_of_data = len(col_desc_val)

	return render_template('load_data.html', tables=tables, titles=titles, dataframes_list=dataframes_list, 
		row=row, col=col, list_of_col=list_of_col, column=column, col_desc_keys=col_desc_keys, 
		col_desc_val=col_desc_val, lod=length_of_data, file=file, mis_col=mis_col, 
		mis_col_val=mis_col_val, len_mis=len_mis)





# redirecting functions for load_data.html
@app.route('/dataframe/<file>', methods=['GET'])
def dataframe_file(file):
	return redirect(url_for('load_data', file=file))


@app.route('/dataframe/<file>/<column>', methods=['GET'])
def dataframe_column(file, column):
	return redirect(url_for('load_data', file=file, column=column))

@app.route('/dataframe/<file>/miss-val-row', methods=['GET', 'POST'])
def miss_val_row(file):
	dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))
	dataFrame = dp.drop_missing_rows(dataFrame)
	dp.save_dataframe(dataFrame,  os.path.join(app.config['UPLOAD_FOLDER'], file))
	# print(file)
	return redirect(url_for('load_data', file=file))

@app.route('/dataframe/<file>/miss-val-column', methods=['GET', 'POST'])
def miss_val_column(file):
	dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))
	dataFrame = dp.drop_missing_cols(dataFrame)
	dp.save_dataframe(dataFrame, os.path.join(app.config['UPLOAD_FOLDER'], file))
	return redirect(url_for('load_data', file=file))


@app.route('/dataframe/<file>/col-list', methods=['GET', 'POST'])
def encode_data(file):
	col_list = request.form.getlist('inputstate')
	dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))
	dataFrame = dp.label_encoder(dataFrame, col_list)
	dp.save_dataframe(dataFrame, os.path.join(app.config['UPLOAD_FOLDER'], file))
	# print(col_list)
	return redirect(url_for('load_data', file=file))

@app.route('/dataframe/<file>/cols-list', methods=['GET', 'POST'])
def drop_columns(file):
	col_list = request.form.getlist('inputState')
	dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))
	dataFrame = dp.drop_cols(dataFrame, col_list)
	dp.save_dataframe(dataFrame, os.path.join(app.config['UPLOAD_FOLDER'], file))
	# print(col_list)
	return redirect(url_for('load_data', file=file))


@app.route('/dataframe/<file>/normalization', methods=['GET', 'POST'])
def normalizers(file):
	normalizer = request.form.get('normalizer')
	#print(str(normalizer))
	dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))
	if normalizer == 'Standard_Scaler':
		dataFrame = dp.standard_(dataFrame)
	elif normalizer == 'Min_Max_Scaler':
		dataFrame = dp.min_max_(dataFrame)
	elif normalizer == 'Robust_Scaler':
		dataFrame = dp.robust_(dataFrame)
	elif normalizer == 'Normalizer':
		dataFrame = dp.normalization_(dataFrame)

	dp.save_dataframe(dataFrame, os.path.join(app.config['UPLOAD_FOLDER'], file))
	return redirect(url_for('load_data', file=file))


@app.route('/data_visual', methods=['GET', 'POST'])
def data_visual():

	plot_name = request.args.get('plot_name')
	file = session['file']

	if file is not None:
		dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))
		(row, col, list_of_col) = dp.data_preparation(dataFrame)
	#fig_name = session['fig_name']
	fig_name = request.args.get('fig')

	return render_template('third_page.html', plot_name=plot_name, fig_name=fig_name, list_of_col=list_of_col)


@app.route('/data_visual/<plot_name>', methods=['GET', 'POST'])
def plot_data(plot_name):
	# print(plot_name)
	return redirect(url_for('data_visual', plot_name=plot_name))


@app.route('/data_visual/<plot_name>/values', methods=['GET', 'POST'])
def visual_data(plot_name):

	x_col = request.form.get('xcolumn')
	y_col = request.form.get('ycolumn')
	hue = request.form.get('hue')
	esti_ = request.form.get('estimator')

	path = app.config['IMAGE_FOLDER']

	file = session['file']

	dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))

	if hue == 'Choose':
		hue = None

	if y_col == 'Choose':
		y_col = None

	if x_col == 'Choose':
		x_col = None


	if plot_name == 'scatter':
		fig_name = dplot.plot_scatter(x_col, y_col, path=path, data=dataFrame, hue=hue)

	elif plot_name == 'line':
		fig_name = dplot.plot_line(x_col, y_col, path=path, data=dataFrame, hue=hue)

	elif plot_name == 'bar':
		fig_name = dplot.plot_bar(x_col, y_col, path=path, data=dataFrame, hue=hue)

	elif plot_name == 'dist':
		fig_name = dplot.plot_dist(x_col, path=path, data=dataFrame, hue=hue)

	elif plot_name == 'box':
		fig_name = dplot.plot_box(x_col, y_col, path=path, data=dataFrame, hue=hue)

	elif plot_name == 'point':
		fig_name = dplot.plot_point(x_col, y_col, path=path, data=dataFrame, hue=hue)

	elif plot_name == 'count':
		fig_name = dplot.plot_count(x_col, path=path, data=dataFrame, hue=hue)

	elif plot_name == 'violin':
		fig_name = dplot.plot_violin(x_col, y_col, path=path, data=dataFrame, hue=hue)

	elif plot_name == 'strip':
		fig_name = dplot.plot_strip(x_col, y_col, path=path, data=dataFrame, hue=hue)

	elif plot_name == 'swarm':
		fig_name = dplot.plot_swarm(x_col, y_col, path=path, data=dataFrame, hue=hue)

	elif plot_name == 'heatmap':
		fig_name = dplot.heatmap(data=dataFrame)

	#session['fig_name'] = fig_name'

	return redirect(url_for('data_visual', plot_name=plot_name, fig=fig_name))





@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
	list_of_col = []

	list_of_dfs = dp.list_all_dataframes(app.config['UPLOAD_FOLDER'])
	file = request.args.get('file')
	if file is not None:
		dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))
		(row, col, list_of_col) = dp.data_preparation(dataFrame)

	
	#label_col = request.args.get('label_col')
	test_size = request.args.get('test_size')
	label_name = request.args.get('label_name')
	shuffle = request.args.get('shuffle')

	#print(label_col)
	print(test_size)
	print(label_name)
	print(shuffle)


	return render_template('six_in_one.html', list_of_dfs=list_of_dfs, file=file, 
		list_of_col=list_of_col)



# redirecting functions for load_data.html
@app.route('/train_model/<file>', methods=['GET'])
def data_file(file):
	return redirect(url_for('train_model', file=file))



# @app.route('/train_model/<file>/<label_col>',methods=['GET', 'POST'])
# def list_col(file, label_col):
# 	return redirect(url_for('train_model', file=file, label_col=label_col))



@app.route('/train_model/<file>/col-list', methods=['GET', 'POST'])
def encode_cols(file):
	col_list = request.form.getlist('inputstate')
	dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))
	dataFrame = dp.label_encoder(dataFrame, col_list)
	dp.save_dataframe(dataFrame, os.path.join(app.config['UPLOAD_FOLDER'], file))
	# print(col_list)
	return redirect(url_for('train_model', file=file))


@app.route('/train_model/<file>/split-data', methods=['GET', 'POST'])
def split_data(file):

	test_size = request.form.get('test_size')
	test_size = float(test_size)

	label_name = request.form.get('label_name')

	if request.form.get("shuffle"):
		shuffle = True

	if test_size == None:
		test_size = 0.25
	if shuffle == None:
		shuffle = True

	# Data Preparation
	data_obj = prepare_data()
	#dataFrame = data_obj.load_dataframe(session['filepath'])
	#dataFrame, n_of_cols, n_of_rows, list_of_col = data_obj.prepare_Dataframe(session['filepath'])
	dataFrame = dp.load_dataframe(os.path.join(app.config['UPLOAD_FOLDER'], file))
	data_dict = data_obj.split_data(dataFrame, test_size, label_name, shuffle, app.config['UPLOAD_FOLDER'])


	session['X_train'] = os.path.join(app.config['UPLOAD_FOLDER'], 'X_train.pkl')
	session['X_test'] = os.path.join(app.config['UPLOAD_FOLDER'], 'X_test.pkl')
	session['y_train'] = os.path.join(app.config['UPLOAD_FOLDER'], 'y_train.pkl')
	session['y_test'] = os.path.join(app.config['UPLOAD_FOLDER'], 'y_test.pkl')	

	return redirect(url_for('train_model', file=file))#, test_size=test_size, label_name=label_name, shuffle=shuffle))







# Training Models

@app.route('/knn-model', methods=['GET', 'POST'])
def knn_model_():

	# Training Data
	X_train = dp.load_dataframe(session['X_train'])
	X_test = dp.load_dataframe(session['X_test'])
	y_train = dp.load_dataframe(session['y_train'])
	y_test = dp.load_dataframe(session['y_test'])


	# Model Parameters
	n_neighbors = request.form.get('n_neighbors')
	weights = request.form.get('weights')
	algorithm = request.form.get('algorithm')
	leaf_size = request.form.get('leaf_size')
	power_param = request.form.get('power_param')

	n_neighbors = int(n_neighbors)
	leaf_size = int(leaf_size)
	power_param = int(power_param)


	# Training the model
	model_obj = knn_model(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, 
			p=power_param)
	model = model_obj.train_model(X_train, y_train, app.config['MODEL_FOLDER'])

	modelpath = os.path.join(app.config['MODEL_FOLDER'], 'knn_model.pkl')

	train_acc = model_obj.model_score(modelpath, X_train, y_train)

	test_acc = model_obj.model_score(modelpath, X_test, y_test)

	y_pred = model_obj.predict_output(modelpath, app.config['UPLOAD_FOLDER'], X_test)
	#y_pred = np.array(y_pred)


	return render_template('final_output.html', train_acc=train_acc, test_acc=test_acc, 
		y_pred=y_pred)




@app.route('/radius-nn-model', methods=['GET', 'POST'])
def radius_nn_model_():
	# Training Data
	X_train = dp.load_dataframe(session['X_train'])
	X_test = dp.load_dataframe(session['X_test'])
	y_train = dp.load_dataframe(session['y_train'])
	y_test = dp.load_dataframe(session['y_test'])



	# Model Parameters
	radius = request.form.get('radius')
	weights = request.form.get('weights')
	algorithm = request.form.get('algorithm')
	leaf_size = request.form.get('leaf_size')
	power_param = request.form.get('power_param')
	outlier_label = request.form.get('outlier_label')

	radius = float(radius)
	leaf_size = int(leaf_size)
	power_param = int(power_param)

	print(algorithm)


	# Training the model
	model_obj = radius_nn_model(radius=radius, weights=weights, 
			algorithm=algorithm, leaf_size=leaf_size, p=power_param, outlier_label=outlier_label)

	model = model_obj.train_model(X_train, y_train, app.config['MODEL_FOLDER'])

	modelpath = os.path.join(app.config['MODEL_FOLDER'], 'radius_nn_model.pkl')

	train_acc = model_obj.model_score(modelpath, X_train, y_train)

	test_acc = model_obj.model_score(modelpath, X_test, y_test)

	y_pred = model_obj.predict_output(modelpath, app.config['UPLOAD_FOLDER'], X_test)
	#y_pred = np.array(y_pred)


	return render_template('final_output.html', train_acc=train_acc, test_acc=test_acc, 
		y_pred=y_pred)









@app.route('/linreg-model', methods=['GET', 'POST'])
def linreg_model_():

	# Training Data
	X_train = dp.load_dataframe(session['X_train'])
	X_test = dp.load_dataframe(session['X_test'])
	y_train = dp.load_dataframe(session['y_train'])
	y_test = dp.load_dataframe(session['y_test'])


	# Model Parameters
	fit_intercept = request.form.get('fit_intercept')
	normalize = request.form.get('normalize')
	copy_X = request.form.get('copy_X')
	n_jobs = request.form.get('n_jobs')

	n_jobs = int(n_jobs)


	# Training the model
	model_obj = linreg_model(fit_intercept=fit_intercept, normalize=normalize, 
			copy_X=copy_X, n_jobs=n_jobs)
	model = model_obj.train_model(X_train, y_train, app.config['MODEL_FOLDER'])

	modelpath = os.path.join(app.config['MODEL_FOLDER'], 'lin_reg_model.pkl')

	train_acc = model_obj.model_score(modelpath, X_train, y_train)

	test_acc = model_obj.model_score(modelpath, X_test, y_test)

	y_pred = model_obj.predict_output(modelpath, app.config['UPLOAD_FOLDER'], X_test)
	#y_pred = np.array(y_pred)

	return render_template('final_output.html', train_acc=train_acc, test_acc=test_acc, y_pred=y_pred)









@app.route('/logreg-model', methods=['GET', 'POST'])
def logreg_model_():

	# Training Data
	X_train = dp.load_dataframe(session['X_train'])
	X_test = dp.load_dataframe(session['X_test'])
	y_train = dp.load_dataframe(session['y_train'])
	y_test = dp.load_dataframe(session['y_test'])


	# Model Parameters
	penalty = request.form.get('penalty')
	dual = request.form.get('dual')
	C = request.form.get('C')
	fit_intercept = request.form.get('fit_intercept')
	solver = request.form.get('solver')
	multi_class = request.form.get('multi_class')

	C = float(C)

	if fit_intercept=="True":
		fit_intercept=True
	elif fit_intercept=="False":
		fit_intercept=False

	if dual=="True":
		dual=True
	elif dual=="False":
		dual=False


	# Training the model
	model_obj = logreg_model(penalty=penalty, dual=dual, C=C, fit_intercept=fit_intercept, 
			solver=solver, multi_class=multi_class)
	model = model_obj.train_model(X_train, y_train, app.config['MODEL_FOLDER'])

	modelpath = os.path.join(app.config['MODEL_FOLDER'], 'logreg_model.pkl')

	train_acc = model_obj.model_score(modelpath, X_train, y_train)

	test_acc = model_obj.model_score(modelpath, X_test, y_test)

	y_pred = model_obj.predict_output(modelpath, app.config['UPLOAD_FOLDER'], X_test)
	#y_pred = np.array(y_pred)


	return render_template('final_output.html', train_acc=train_acc, test_acc=test_acc, y_pred=y_pred)







@app.route('/svc-model', methods=['GET', 'POST'])
def svc_model_():

	# Training Data
	X_train = dp.load_dataframe(session['X_train'])
	X_test = dp.load_dataframe(session['X_test'])
	y_train = dp.load_dataframe(session['y_train'])
	y_test = dp.load_dataframe(session['y_test'])


	# Model Parameters
	C = request.form.get('C')
	kernel = request.form.get('kernel')
	degree = request.form.get('degree')
	gamma = request.form.get('gamma')
	decision_function_shape = request.form.get('decision_function_shape')

	C = float(C)
	degree = int(degree)


	# Training the model
	model_obj = svm_model(C=C, kernel=kernel, degree=degree, gamma=gamma, decision_function_shape=decision_function_shape)
	model = model_obj.train_model(X_train, y_train, app.config['MODEL_FOLDER'])

	modelpath = os.path.join(app.config['MODEL_FOLDER'], 'svm_model.pkl')

	train_acc = model_obj.model_score(modelpath, X_train, y_train)

	test_acc = model_obj.model_score(modelpath, X_test, y_test)

	y_pred = model_obj.predict_output(modelpath, app.config['UPLOAD_FOLDER'], X_test)
	#y_pred = np.array(y_pred)


	return render_template('final_output.html', train_acc=train_acc, test_acc=test_acc, y_pred=y_pred)










@app.route('/linsvc-model', methods=['GET', 'POST'])
def linsvc_model():

	# Training Data
	X_train = dp.load_dataframe(session['X_train'])
	X_test = dp.load_dataframe(session['X_test'])
	y_train = dp.load_dataframe(session['y_train'])
	y_test = dp.load_dataframe(session['y_test'])


	# Model Parameters
	penalty = request.form.get('penalty')
	loss = request.form.get('loss')
	dual = request.form.get('dual')
	C = request.form.get('C')
	multi_class = request.form.get('multi_class')

	C = float(C)

	if dual=="True":
		dual=True
	elif dual=="False":
		dual=False


	# Training the model
	model_obj = linear_svc_model(penalty=penalty, loss=loss, dual=dual, C=C, multi_class=multi_class)
	model = model_obj.train_model(X_train, y_train, app.config['MODEL_FOLDER'])

	modelpath = os.path.join(app.config['MODEL_FOLDER'], 'linsvc_model.pkl')

	train_acc = model_obj.model_score(modelpath, X_train, y_train)

	test_acc = model_obj.model_score(modelpath, X_test, y_test)

	y_pred = model_obj.predict_output(modelpath, app.config['UPLOAD_FOLDER'], X_test)
	#y_pred = np.array(y_pred)


	return render_template('final_output.html',  train_acc=train_acc, test_acc=test_acc, y_pred=y_pred)





@app.route('/gauss-nb-model', methods=['GET', 'POST'])
def gaussian_nb_model_():

	# Training Data
	X_train = dp.load_dataframe(session['X_train'])
	X_test = dp.load_dataframe(session['X_test'])
	y_train = dp.load_dataframe(session['y_train'])
	y_test = dp.load_dataframe(session['y_test'])


	# Training the model
	model_obj = gaussian_nb_model()
	model = model_obj.train_model(X_train, y_train, app.config['MODEL_FOLDER'])

	modelpath = os.path.join(app.config['MODEL_FOLDER'], 'gauss_nb_model.pkl')

	train_acc = model_obj.model_score(modelpath, X_train, y_train)

	test_acc = model_obj.model_score(modelpath, X_test, y_test)

	y_pred = model_obj.predict_output(modelpath, app.config['UPLOAD_FOLDER'], X_test)
	#y_pred = np.array(y_pred)


	return render_template('final_output.html', train_acc=train_acc, test_acc=test_acc, y_pred=y_pred)









@app.route('/multi-nb-model', methods=['GET', 'POST'])
def multinomial_nb_model_():

	# Training Data
	X_train = dp.load_dataframe(session['X_train'])
	X_test = dp.load_dataframe(session['X_test'])
	y_train = dp.load_dataframe(session['y_train'])
	y_test = dp.load_dataframe(session['y_test'])


	# Training the model
	model_obj = multinomial_nb_model()
	model = model_obj.train_model(X_train, y_train, app.config['MODEL_FOLDER'])

	modelpath = os.path.join(app.config['MODEL_FOLDER'], 'multi_nb_model.pkl')

	train_acc = model_obj.model_score(modelpath, X_train, y_train)

	test_acc = model_obj.model_score(modelpath, X_test, y_test)

	y_pred = model_obj.predict_output(modelpath, app.config['UPLOAD_FOLDER'], X_test)
	#y_pred = np.array(y_pred)


	return render_template('final_output.html', train_acc=train_acc, test_acc=test_acc, y_pred=y_pred)







@app.route('/bern-nb-model', methods=['GET', 'POST'])
def bernoulli_nb_model_():

	# Training Data
	X_train = dp.load_dataframe(session['X_train'])
	X_test = dp.load_dataframe(session['X_test'])
	y_train = dp.load_dataframe(session['y_train'])
	y_test = dp.load_dataframe(session['y_test'])



	# Training the model
	model_obj = bernoulli_nb_model()
	model = model_obj.train_model(X_train, y_train, app.config['MODEL_FOLDER'])

	modelpath = os.path.join(app.config['MODEL_FOLDER'], 'bern_nb_model.pkl')

	train_acc = model_obj.model_score(modelpath, X_train, y_train)

	test_acc = model_obj.model_score(modelpath, X_test, y_test)

	y_pred = model_obj.predict_output(modelpath, app.config['UPLOAD_FOLDER'], X_test)
	#y_pred = np.array(y_pred)


	return render_template('final_output.html', train_acc=train_acc, test_acc=test_acc, y_pred=y_pred)












@app.route('/dec-tree-model', methods=['GET', 'POST'])
def decision_tree_model_():

	# Training Data
	X_train = dp.load_dataframe(session['X_train'])
	X_test = dp.load_dataframe(session['X_test'])
	y_train = dp.load_dataframe(session['y_train'])
	y_test = dp.load_dataframe(session['y_test'])


	# Model Parameters
	criterion = request.form.get('criterion')
	splitter = request.form.get('splitter')
	max_depth = request.form.get('max_depth')
	max_features = request.form.get('max_features')

	if max_depth == "-1":
		max_depth = None
	else:
		max_depth = int(max_depth)

	if max_features == "None":
		max_features = None


	# Training the model
	model_obj = decision_tree_model(criterion=criterion, splitter=splitter, 
			max_depth=max_depth, max_features=max_features)
	model = model_obj.train_model(X_train, y_train, app.config['MODEL_FOLDER'])

	modelpath = os.path.join(app.config['MODEL_FOLDER'], 'dec_tree_model.pkl')

	train_acc = model_obj.model_score(modelpath, X_train, y_train)

	test_acc = model_obj.model_score(modelpath, X_test, y_test)

	y_pred = model_obj.predict_output(modelpath, app.config['UPLOAD_FOLDER'], X_test)
	#y_pred = np.array(y_pred)


	return render_template('final_output.html', train_acc=train_acc, test_acc=test_acc, y_pred=y_pred)







@app.route('/rand-forest-model', methods=['GET', 'POST'])
def random_forest_model_():

	# Training Data
	X_train = dp.load_dataframe(session['X_train'])
	X_test = dp.load_dataframe(session['X_test'])
	y_train = dp.load_dataframe(session['y_train'])
	y_test = dp.load_dataframe(session['y_test'])


	# Model Parameters
	n_estimators = request.form.get('n_estimators')
	criterion = request.form.get('criterion')
	max_depth = request.form.get('max_depth')
	max_features = request.form.get('max_features')

	n_estimators = int(n_estimators)

	if max_depth == "-1":
		max_depth = None
	else:
		max_depth = int(max_depth)

	if max_features == "None":
		max_features = None

	# Training the model
	model_obj = random_forest_model(n_estimators=n_estimators, criterion=criterion, 
			max_depth=max_depth, max_features=max_features)
	model = model_obj.train_model(X_train, y_train, app.config['MODEL_FOLDER'])

	modelpath = os.path.join(app.config['MODEL_FOLDER'], 'rf_model.pkl')

	train_acc = model_obj.model_score(modelpath, X_train, y_train)

	test_acc = model_obj.model_score(modelpath, X_test, y_test)

	y_pred = model_obj.predict_output(modelpath, app.config['UPLOAD_FOLDER'], X_test)
	#y_pred = np.array(y_pred)


	return render_template('final_output.html', train_acc=train_acc, test_acc=test_acc, 
	y_pred=y_pred)








# running the application
if __name__ == '__main__':
	app.run(debug=True)