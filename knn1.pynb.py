#-------------------------------------------------------------------------
# AUTHOR: Ajith Elumalai
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #4
# TIME SPENT: 1 day
#-----------------------------------------------------------*/import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

# Defining the hyperparameter values of KNN
k_values = [ 15]  # Example: Trying specific k values
p_values = [ 1]    # Example: Including p=3
w_values = ['uniform', 'distance']

# Reading the training data
train_data = pd.read_csv('weather_test.csv')  
test_data = pd.read_csv('weather_test.csv')

# Assuming the last column is the class label
X_training = train_data.iloc[:, 1:-1].values  # All columns except the first and last
y_training = train_data.iloc[:, -1].values   # Last column (target)

# Discretizing the target values into 11 discrete classes
y_training = np.digitize(y_training, classes) - 1  # Make classes index-based

# Reading the test data
X_test = test_data.iloc[:, 1:-1].values  # All columns except the first and last
y_test = test_data.iloc[:, -1].values

# Discretizing the target values in the test set
y_test = np.digitize(y_test, classes) - 1  # Make classes index-based

# Initialize the variable to track highest accuracy
highest_accuracy = 0
best_params = (0, 0, '')

# Grid Search over k, p, and w
for k in k_values:
    for p in p_values:
        for w in w_values:
            
            # Fitting the KNN model
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf.fit(X_training, y_training)
            
            # Initialize the count for correct predictions
            correct_predictions = 0
            
            # Make predictions and calculate accuracy
            for x_test_sample, y_test_sample in zip(X_test, y_test):
                prediction = clf.predict([x_test_sample])[0]
                
                # Calculate the percentage difference
                percentage_diff = 100 * abs(prediction - y_test_sample) / y_test_sample
                
                # Check if prediction is within the acceptable range
                if percentage_diff <= 15:
                    correct_predictions += 1
            
            # Calculate the current accuracy
            accuracy = correct_predictions / len(y_test)
            
            # Check if this accuracy is the highest
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_params = (k, p, w)
                print(f"Highest KNN accuracy so far: {highest_accuracy:.2f} Parameters: k = {k}, p = {p}, weight = {w}")