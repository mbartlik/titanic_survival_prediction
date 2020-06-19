from sklearn.ensemble import RandomForestClassifier
from titanic_data_loader import titanic_data_loader
from model_evaluation import evaluate_model_accuracy

def random_forest_generator(n_estimators, max_depth, min_samples_split, min_samples_leaf, data_split=False, data_split_point=700):

	# Instantiate model
	model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, criterion='gini')

	# Load data - data_split point determines how the train/test data is split
	if data_split:
		X_train, y_train, X_test, y_test = titanic_data_loader(data_split=data_split, data_split_point=data_split_point)
	else:
		X_train, y_train = titanic_data_loader(data_split=data_split, data_split_point=data_split_point)

	# Fit the model
	model.fit(X_train, y_train)

	return model




# Desired hyperparameters to test
n_estimators = [90,100,110]
max_depth = [8,10,12]
min_samples_splits = [5,6,7]
min_samples_leafs = [1]


results = []

# Loop through each hyperparameter
for estimators in n_estimators:
	for depth in max_depth:
		for min_samples_split in min_samples_splits:
			for min_samples_leaf in min_samples_leafs:

				# Will be sum of accuracies for each data split given these hyperparameters
				running_accuracy = 0

				for data_split_point in [0,100,200,300,400,500,600,700]:
					# Instantiate model on these hyperparamaters
					model = random_forest_generator(estimators, depth, min_samples_split, min_samples_leaf, data_split=True, data_split_point=data_split_point)

					running_accuracy += evaluate_model_accuracy(model, data_split_point)

				# Create dictionary of hyperparameters and accuracy
				results_dict = {'n_estimators': estimators,
								'max_depth': depth,
								'min_samples_split': min_samples_split,
								'min_samples_leaf': min_samples_leaf,
								'Accuracy': running_accuracy*100.0/8
								}
				# Add to results
				results.append(results_dict)

# Output results
for result in results:
	print(result)




"""
n_estimators = [20,40,60,80,100,120,140]
max_depth = [None,5,10,15,20]
min_samples_splits = [2,4,6,8]
min_samples_leafs = [1,3,5]
{'n_estimators': 20, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 1, 'Accuracy': 83.1151832460733}
{'n_estimators': 40, 'max_depth': 15, 'min_samples_split': 8, 'min_samples_leaf': 1, 'Accuracy': 83.44240837696336}
{'n_estimators': 40, 'max_depth': 15, 'min_samples_split': 8, 'min_samples_leaf': 3, 'Accuracy': 83.1151832460733}
{'n_estimators': 40, 'max_depth': 20, 'min_samples_split': 8, 'min_samples_leaf': 1, 'Accuracy': 83.18062827225131}
{'n_estimators': 60, 'max_depth': None, 'min_samples_split': 8, 'min_samples_leaf': 1, 'Accuracy': 83.18062827225131}
{'n_estimators': 60, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 3, 'Accuracy': 83.1151832460733}
{'n_estimators': 60, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 1, 'Accuracy': 83.31151832460732}
{'n_estimators': 80, 'max_depth': 10, 'min_samples_split': 6, 'min_samples_leaf': 1, 'Accuracy': 83.31151832460732}
{'n_estimators': 100, 'max_depth': None, 'min_samples_split': 6, 'min_samples_leaf': 1, 'Accuracy': 83.1151832460733}
*****{'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 6, 'min_samples_leaf': 1, 'Accuracy': 83.76963350785338}
{'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 1, 'Accuracy': 83.1806282722513}
{'n_estimators': 140, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 1, 'Accuracy': 83.37696335078532}













"""



