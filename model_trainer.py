from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from titanic_data_loader import titanic_data_loader

def model_trainer(model_type,data_split=False, data_split_point=700):

	# Instantiate model based on input
	if model_type == 'logistic_regression':
		model = LogisticRegression()
	elif model_type == 'decision_tree':
		model = DecisionTreeClassifier()
	elif model_type == 'naive_bayes':
		model = GaussianNB()
	elif model_type == 'SVC':
		model = SVC()
	elif model_type == 'random_forest':
		model = RandomForestClassifier()
	elif model_type == 'ada_boost':
		model = AdaBoostClassifier()

	# Load data - data_split point determines how the train/test data is split
	if data_split:
		X_train, y_train, X_test, y_test = titanic_data_loader(data_split=data_split, data_split_point=data_split_point)
	else:
		X_train, y_train = titanic_data_loader(data_split=data_split, data_split_point=data_split_point)

	# Fit the model
	model.fit(X_train, y_train)

	return model

