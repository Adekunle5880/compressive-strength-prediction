import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Load the csv file
df = pd.read_csv("compressive_strenght_prediction_data.csv")

# Select independent variable and dependent variable
x = df[['Age (days)', 'weight of fine aggregrate', 'weight of cement', 'percentage of Termite Mound Concrete (TMD)', 
        'weight of coarse aggregate', 'weight of Termite Mound Concrete (TMD)', 'compaction count', 'slump result',
        'Density of Cube']]
y = df['compressive strength']

# Split the dataset into train and test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

# Standardize the training set and save the scaler
stand = StandardScaler()
Fit = stand.fit(xtrain)
pickle.dump(Fit, open("scaler.pkl", "wb"))
xtrain_scl = Fit.transform(xtrain)
xtest_scl = Fit.transform(xtest)

# Train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(xtrain_scl, ytrain)

# Evaluate the model's performance
score = rf.score(xtest_scl, ytest)
print('R2 Score:', score)
y_predict = rf.predict(xtest_scl)
mse = mean_squared_error(ytest, y_predict)
print('Mean Squared Error:', mse)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

# Plot the predicted vs actual values
plt.figure(figsize=[17, 8])
plt.scatter(y_predict, ytest)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the trained Random Forest model
pickle.dump(rf, open("random_forest_model_new.pkl", "wb"))
