#-------------------------------------------------------------------------
# AUTHOR: Ajith Elumalai
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes classifier with grid search on smoothing parameter
# FOR: CS 5990- Assignment #4
# TIME SPENT: 1 day
#-------------------------------------------------------------------------

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

# 11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

# reading the training data
train_df = pd.read_csv('weather_training.csv')

# update the training class values according to the discretization (11 values only)
X_training = train_df.select_dtypes(include=[np.number]).iloc[:, :-1].values
y_training = train_df.select_dtypes(include=[np.number]).iloc[:, -1].values
y_training = np.digitize(y_training, classes) - 1

# reading the test data
test_df = pd.read_csv('weather_test.csv')

# update the test class values according to the discretization (11 values only)
X_test = test_df.select_dtypes(include=[np.number]).iloc[:, :-1].values
y_test = test_df.select_dtypes(include=[np.number]).iloc[:, -1].values
y_test = np.digitize(y_test, classes) - 1

# loop over the hyperparameter value (s)
highest_accuracy = 0
best_s = None

for s in s_values:

    # fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, y_training)

    # make the naive_bayes prediction for each test sample and start computing its accuracy
    correct_predictions = 0
    for x_test_sample, y_test_sample in zip(X_test, y_test):
        prediction = clf.predict([x_test_sample])[0]

        if y_test_sample != 0:
            percent_diff = 100 * abs(prediction - y_test_sample) / abs(y_test_sample)
        else:
            percent_diff = float('inf')

        if percent_diff <= 15:
            correct_predictions += 1

    accuracy = correct_predictions / len(y_test)

    # check if the calculated accuracy is higher than the previously one calculated
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_s = s
        print(f"Highest Naive Bayes accuracy so far: {highest_accuracy:.2f} Parameter: s = {best_s:.10f}")


