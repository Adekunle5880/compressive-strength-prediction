import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#import pickle


# Load the csv file
df = pd.read_csv("compressive_strenght_prediction_data.csv") 

print(df.head())

# Select independent variable and dependent variable
x = df[['Age (days)', 'weight of fine aggregrate', 'weight of cement', 'percentage of Termite Mound Concrete (TMD)', 'weight of coarse aggregate', 'weight of Termite Mound Concrete (TMD)', 'compaction count', 'slump result', 
        'Density of Cube']]
# dependent variables
y = df['compressive strength']

# Split the dataset into train and test
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=42)

stand = StandardScaler()
Fit = stand.fit(xtrain)
xtrain_scl = Fit.transform(xtrain)
xtest_scl = Fit.transform(xtest)

# Evaluate the model
lr=LinearRegression()
fit=lr.fit(xtrain_scl,ytrain)
score = lr.score(xtest_scl,ytest)
print('predcted score is : {}'.format(score))
print('..................................')
y_predict = lr.predict(xtest_scl)
print('mean_sqrd_error is ==',mean_squared_error(ytest,y_predict))
rms = np.sqrt(mean_squared_error(ytest,y_predict)) 
print('root mean squared error is == {}'.format(rms))

plt.figure(figsize=[17,8])
plt.scatter(y_predict,ytest)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red')
plt.xlabel('predicted')
plt.ylabel('orignal')
plt.show()

# Make the Pickle file of our model
#pickle.dump(lr, open("model.pkl", "wb"))

# Get the coefficients and intercept from the trained model
coefficients = lr.coef_
intercept = lr.intercept_

# List of feature names
feature_names = x.columns

# Create a list of terms in the regression formula
terms = [f"{round(coefficients[i], 2)}*{feature_names[i]}" for i in range(len(coefficients))]

# Combine terms with '+' and add the intercept
formula = f"{round(intercept, 2)} + " + " + ".join(terms)

# Display the regression formula
print("Regression Formula:")
print("y =", formula)
