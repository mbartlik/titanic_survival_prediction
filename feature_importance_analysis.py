import pandas as pd
from model_trainer import model_trainer
from titanic_data_loader import titanic_data_loader

# Train model and get importance values
model = model_trainer('random_forest')
importances = model.feature_importances_

# Load data to obtain feature names
survival_data,survival = titanic_data_loader()

# Create and print results data frame
results = pd.DataFrame({'feature':survival_data.columns, 'importance': importances})
print(results)