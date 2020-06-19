from model_trainer import model_trainer
from titanic_data_loader import titanic_data_loader

def evaluate_model_accuracy(model,data_split_point):

	# Load data
	X_train, y_train, X_test, y_test = titanic_data_loader(data_split=True,data_split_point=data_split_point)

	# Make predictions on test data
	predictions = model.predict(X_test)

	# Compare to the test survival values
	number_correct = 0
	for i in range(len(predictions)):
		if predictions[i] == y_test[i+data_split_point]:
			number_correct += 1

	# Calculate accuracy
	accuracy = number_correct*1.0/len(predictions)

	return accuracy



model_names = ['logistic_regression','naive_bayes','decision_tree','SVC','random_forest','ada_boost']
data_split_points = [0,100,200,300,400,500,600,700]
accuracies = []

# For each model type create and train a model on different data points
# Calculate accuracy on the corresponding test set 
for model_name in model_names:

	# Accuracy total for this model
	running_accuracy = 0

	for data_split_point in data_split_points:

		# Instantiate model
		model = model_trainer(model_name,data_split=True, data_split_point=data_split_point)

		# Add to accuracy total for this model
		running_accuracy += evaluate_model_accuracy(model,data_split_point)

	# Calculate average accuracy and add to accuracies list
	accuracies.append(running_accuracy/len(data_split_points))


# Print accuracies
for accuracy in accuracies:
	print("Accuracy: {0:.2f}%".format(accuracy*100))

"""
Output of this code:
Accuracy: 79.48%
Accuracy: 75.81%
Accuracy: 79.06%
Accuracy: 66.39%
Accuracy: 80.21%
Accuracy: 80.94%

This means adaBoost is the best model to use
"""
