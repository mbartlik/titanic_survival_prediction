# Titanic Survival Predictor

## Description
This project generates a model based off of information about the passengers of the titanic, and the model can predict, given a number of attributes, whether or not it is likely that the given passenger would survive the titanic. These attributes include things such as gender, age, type of cabin, and where they embarked from. This project is a challenge made by Kaggle. 

## Random Forest
I found that a random forest classifier was best used for this project. By running predict_kaggle_data.py, the program makes a random forest model off of the training data, processes test data, and then makes and saves predictions in the results folder. 
