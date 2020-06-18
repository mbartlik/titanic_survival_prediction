import pandas as pd
from model_trainer import model_trainer

# Model is to be trained on ALL of the given labeled data
# AdaBoost will be used as this was determined to be the best in model_evaluation
# Predictions will be made on the Kaggle test data


test_data = pd.read_csv('data/test.csv')


# One hot encoding - pclass, embarked
one_hot_pclass = pd.get_dummies(test_data['Pclass'])
one_hot_embarked = pd.get_dummies(test_data['Embarked'])

# Drop the original categorical columns
test_data = test_data.drop(['Pclass','Embarked'], axis=1)

# Add the one-hot columns
test_data = test_data.join(one_hot_pclass)
test_data = test_data.join(one_hot_embarked)

# Drop the names (passenger id is all that is necessary) and ticket numbers and cabin
test_data = test_data.drop('Name', axis=1)
test_data = test_data.drop('Ticket', axis=1)
test_data = test_data.drop('Cabin', axis=1)

# Make male/female 1s and 0s
test_data['Sex'].loc[test_data['Sex'] == 'female'] = 1
test_data['Sex'].loc[test_data['Sex'] != 1] = 0

# Replace null values with the mean of that feature - it is only age and fare
test_data['Age'].loc[test_data['Age'].isnull()] = test_data['Age'].mean()
test_data['Fare'].loc[test_data['Fare'].isnull()] = test_data['Fare'].mean()

# Separate passenger id column from the data
passenger_id = test_data['PassengerId']
test_data = test_data.drop('PassengerId', axis=1)

# Train ada boost model
ada_boost_model = model_trainer('ada_boost')

# Make predictions
predictions = ada_boost_model.predict(test_data)

# Combine passenger id with predictions as output
passenger_id = pd.Series.to_numpy(passenger_id)
results = pd.DataFrame()
results['PassengerId'] = passenger_id
results['Survived'] = predictions
print(results)

# Save results
results.to_csv('results/results.csv', index=False)
