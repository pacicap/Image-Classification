#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 19:00:34 2023

@author: pacicap
"""

import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the data from Images.csv
images = []
labels = []
with open('Images.csv') as f:
    reader = csv.reader(f, delimiter=';')
    next(reader) # skip the first row (number of images)
    for row in reader:
        image_id = row[0]
        image_class = row[1]
        images.append(image_id)
        labels.append(image_class)
        

# Load the data from EdgeHistogram.csv
features = []
with open('EdgeHistogram.csv') as f:
    reader = csv.reader(f, delimiter=';')
    header = next(reader)
    num_images = int(header[0])
    num_dimensions = int(header[1])
    for row in reader:
        image_id = row[0]
        feature_vector = [float(x) for x in row[1:]]
        features.append(feature_vector)

# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)
features = np.array(features)

accuracies = []

# Specify the number of samples per split and the number of splits to generate
n_samples = [5,10,15]

# Train the SVM classifier for each number of samples
for n in n_samples:
# Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

    # Initialize the SVM classifier
    clf = SVC()

    # Hyperparameters to tune
    param_grid = {'C': [0.01, 0.1, 1, 10],
                  'kernel': ['linear', 'rbf', 'sigmoid'], 'gamma': ['auto', 'scale']}

    # Grid search to find the best hyperparameters
    grid_search = GridSearchCV(clf, param_grid, cv=2)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best hyperparameters: ", grid_search.best_params_)
    print()

    # Train the SVM classifier with the best hyperparameters
    clf = grid_search.best_estimator_ 
    scores = grid_search.best_score_
    clf.fit(X_train, y_train)



    #Evaluate the classifier on the test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print("Number of samples per class:", n)
    print("Accuracy:", accuracy)
    print()
    print()
    print("Accuracy on unknown data is",classification_report(y_test, y_pred))
    print()
    print()
    print(pd.DataFrame({'original' : y_test,'predicted' : y_pred}))
    print("=" * 20)
    

plt.plot(n_samples, accuracies, 'o-')
plt.xlabel('Number of samples per class')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of samples per class')
plt.show()
