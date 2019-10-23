import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from sklearn import preprocessing

def cvScores(model, x_train, y_train, cv):
	"""
	Calculates the cross-validation score for a model.
	
	Parameters
	----------
	model : Machine-Learning model
		Model to be assessed.
	x_train : DataFrame
		Data frame holding the features to be used as training
	y_train : Array
		Array holding the target of the model.
	cv : int
		Number of folds for the cross-validation process.
		
	Returns
	-------
	DataFrame
		Data frame holding the mean of the calculated cross-validation scores and its standard deviation.
	"""
	
	modelCVScores = cross_val_score(model, x_train, y_train, cv = cv, n_jobs = -1)
	modelCVScoresMean = modelCVScores.mean()
	modelCVScoresStd = modelCVScores.std()
	
	return pd.DataFrame({'CV-Mean' : [modelCVScoresMean], 'CV-Sd' : [modelCVScoresStd]})
	
def classif_predict_metric(model, x_test, y_test) :
	"""
	Perform a classification prediction and returns the accuracy and kappa values.
	
	Parameters
	----------
	model : Classification model
		Classification model to be assessed.
	x_test : DataFrame
		Data frame holding the features to be used as testing
	y_test : Array
		Array holding the target of the model.
	
	Returns
	-------
	Tuple
		First element: Array holding the predictions. Second element: Accuracy Score. Third element: Kappa Score
	"""
	modelPredictions = model.predict(x_test)
	
	accuracy_model = accuracy_score(y_test, modelPredictions)
	kappa_model = cohen_kappa_score(y_test, modelPredictions)
	
	return modelPredictions, accuracy_model, kappa_model
	
def featureSelection(data, features_to_remove):
	"""
	Removes features from a data frame.
	
	Parameters
	----------
	data : DataFrame
		Data frame
	features_to_remove : Array
		String array with the names of the features to remove from the data frame
		
	Returns
	DataFrame
		A data frame without the specified features.
	"""
	
	data = data.drop(features_to_remove, axis = 1)
	return data

def featureStandardization(data):
	"""
	Perform a standardization in a data frame. Scale and center.
	
	Parameters
	----------
	data : DataFrame
		Data frame to be standardized.
		
	Returns
	-------
	DataFrame
		A data frame with all its features scaled and centered.
	"""
	
	scaler = preprocessing.StandardScaler().fit(data)
	
	x_std = scaler.transform(data)
	
	return x_std
	
def performRandomSearch(model, x, y, params, n_iter = 100, cv = 3, verbose = 2, random_state = 123, n_jobs = -1, scoring = None):
	"""
	Perform a hyperparamenter random search taking a sample of all the possible combinations among the parameters specified.
	The parameters with the best performance are returned.
	
	Parameters
	----------
	model : Machine-Learning model
		Model to be assessed
		
	x : DataFrame
		Data frame holding the features to be used as training
		
	y : Array
		Array holding the target of the model.
	
	params: dict
		Dictionary with parameters names (string) as keys and distributions
    or lists of parameters to try. Distributions must provide a ``rvs``
    method for sampling (such as those from scipy.stats.distributions).
    If a list is given, it is sampled uniformly.
	
	n_iter : int
		Number of parameter settings that are sampled. n_iter trades
    off runtime vs quality of the solution.
	
	cv : int
		specify the number of folds in the cross-validation.
	
	verbose : int
		Controls the verbosity: the higher, the more messages.
	
	random_state : int
		Pseudo random number generator state used for random uniform sampling
    from lists of possible values instead of scipy.stats distributions.
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.
	
	n_jobs : int
		Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors.
	
	scoring : string, callable, list/tuple, dict or None, default: None
		A single string (see :ref:`scoring_parameter`) or a callable
    (see :ref:`scoring`) to evaluate the predictions on the test set.

    For evaluating multiple metrics, either give a list of (unique) strings
    or a dict with names as keys and callables as values.

    NOTE that when using custom scorers, each scorer should return a single
    value. Metric functions returning a list/array of values can be wrapped
    into multiple scorers that return one value each.

    See :ref:`multimetric_grid_search` for an example.

    If None, the estimator's score method is used.
	
	Returns
	-------
	Dict
		Dict with the best parameters.
	"""
	
	model_random = RandomizedSearchCV(estimator = model, 
									param_distributions = params, 
									n_iter = n_iter, 
									cv = cv, 
									verbose = verbose, 
									random_state = random_state, 
									n_jobs = n_jobs,
									scoring = scoring)
	# Fit the random search model
	model_random.fit(x, y)
	return model_random.best_params_
	
def performGridSearch(model, x, y, params, cv = 3, verbose = 2, n_jobs = -1, scoring = None):
	"""
	Perform a hyperparamenter search testing all the possible combinations among the parameters specified.
	The parameters with the best performance are returned.
	
	model : Machine-Learning model
		Model to be assessed.
		
	x : DataFrame
		Data frame holding the features to be used as training
	
	y : Array
		Array holding the target of the model.
		
	params : dict
		Dictionary with parameters names (string) as keys and distributions
    or lists of parameters to try. Distributions must provide a ``rvs``
    method for sampling (such as those from scipy.stats.distributions).
    If a list is given, it is sampled uniformly.
		
	cv : int
		specify the number of folds in the cross-validation.
		
	verbose : int
		Controls the verbosity: the higher, the more messages.
		
	n_jobs : int
		Number of jobs to run in parallel.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors.
	
	scoring : string, callable, list/tuple, dict or None, default: None
		A single string (see :ref:`scoring_parameter`) or a callable
    (see :ref:`scoring`) to evaluate the predictions on the test set.

    For evaluating multiple metrics, either give a list of (unique) strings
    or a dict with names as keys and callables as values.

    NOTE that when using custom scorers, each scorer should return a single
    value. Metric functions returning a list/array of values can be wrapped
    into multiple scorers that return one value each.

    See :ref:`multimetric_grid_search` for an example.

    If None, the estimator's score method is used.
	
	Returns
	-------
	Dict
		Dict with the best parameters.
	"""
	
	model_random = GridSearchCV(estimator = model, 
								param_grid = params,  
								cv = cv, 
								verbose = verbose, 
								n_jobs = n_jobs,
								scoring = scoring)
	# Fit the random search model
	model_random.fit(x, y)
	return model_random.best_params_