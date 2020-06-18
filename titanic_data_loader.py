import pandas as pd

def titanic_data_loader(data_split=False, data_split_point=700):

	# Data split point is the index that the test data starts from and will continues for 192 values
	if data_split_point > 700:
		data_split_point = 700
	if data_split_point < 0:
		data_split_point = 0

	survival_data = pd.read_csv('data/train.csv')


	# One hot encoding - pclass, embarked
	one_hot_pclass = pd.get_dummies(survival_data['Pclass'])
	one_hot_embarked = pd.get_dummies(survival_data['Embarked'])

	# Drop the original categorical columns
	survival_data = survival_data.drop(['Pclass','Embarked'], axis=1)

	# Add the one-hot columns
	survival_data = survival_data.join(one_hot_pclass)
	survival_data = survival_data.join(one_hot_embarked)

	# Drop the names (passenger id is all that is necessary) and ticket numbers and cabin
	survival_data = survival_data.drop('Name', axis=1)
	survival_data = survival_data.drop('Ticket', axis=1)
	survival_data = survival_data.drop('Cabin', axis=1)

	# Make male/female 1s and 0s
	survival_data['Sex'].loc[survival_data['Sex'] == 'female'] = 1
	survival_data['Sex'].loc[survival_data['Sex'] != 1] = 0

	# Age is the only column with null values - change these to the mean age
	survival_data['Age'].loc[survival_data['Age'].isnull()] = survival_data['Age'].mean()

	# Separate survival column from the data
	survival = survival_data['Survived']
	survival_data = survival_data.drop('Survived', axis=1)
	survival_data = survival_data.drop('PassengerId', axis=1)

	# If the data is to be split then stablish training and testing set
	if data_split:
		X_test = survival_data.iloc[data_split_point:data_split_point+191,:]
		y_test = survival.iloc[data_split_point:data_split_point+191]
		X_train = survival_data.drop(survival_data.index[data_split_point:data_split_point+191])
		y_train = survival.drop(survival.index[data_split_point:data_split_point+191])
		return X_train, y_train, X_test, y_test

	#If we aren't splitting data then just return the X and y as whole
	else:
		return survival_data, survival



	

