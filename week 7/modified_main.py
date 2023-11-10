from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import plotly.io as pio
from scipy.io import arff
import pandas as pd
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate_semi_supervised_rf(X_labeled, y_labeled, X_unlabeled, X_test, y_test, i, perc):
    print(f"running test iteration {i} with percentage {perc}")
    
    # Train the Random Forest classifier on the labeled data
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_labeled, y_labeled)

    # Predict labels for the unlabeled data
    pseudo_labels = rf_classifier.predict(X_unlabeled)

    # Combine the labeled data and pseudo-labeled data
    X_combined = np.vstack((X_labeled, X_unlabeled))
    y_combined = np.concatenate((y_labeled, pseudo_labels))

    # Re-train the Random Forest classifier on the combined dataset
    rf_classifier.fit(X_combined, y_combined)

    # Evaluate the re-trained model on the test set
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy
