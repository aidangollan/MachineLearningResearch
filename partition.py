from sklearn.preprocessing import StandardScaler
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import random

def partition(training_function, is_semi_supervised, avg_amt, path, percentages, name, print_level):
    # Load the data
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    # Convert byte strings to regular strings and encode labels
    df['Class'] = df['Class'].str.decode('utf-8')
    label_encoder = LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['Class'])

    # Prepare the features and labels
    X = df.drop(columns='Class')
    y = df['Class']

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    avg_accuracies = []

    print(f"Starting training with model {name}") if print_level > 0 else None

    for perc in percentages:
        accuracies = []

        for iteration in range(avg_amt):
            print(f"Running percentage {perc} with model {name} on iteration {iteration}") if print_level > 0 else None
            train_proportion = perc / 100.0

            # Split X_train into labeled and unlabeled_temp based on perc
            if train_proportion != 1.0:
                X_labeled, X_unlabeled_temp, y_labeled, _ = train_test_split(X_train, y_train, train_size=train_proportion, random_state=random.randint(0, 100))
            else:
                X_labeled, X_unlabeled_temp, y_labeled = X_train, np.array([]), y_train

            # For semi-supervised, use unlabeled_temp as X_unlabeled
            if is_semi_supervised:
                X_unlabeled = X_unlabeled_temp
            else:
                X_unlabeled = np.array([])  # Empty array for consistency

            accuracy = training_function(X_labeled, y_labeled, X_unlabeled, X_test, y_test, iteration, perc, name, print_level)
            accuracies.append(accuracy)

        avg_accuracies.append(np.mean(accuracies))

    return percentages, avg_accuracies