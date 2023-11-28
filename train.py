from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np

def train_and_evaluate(model: BaseEstimator, X_labeled, y_labeled, X_unlabeled, X_test, y_test, i, perc, name, print_level):
    print(f"Running test iteration {i} on model {name} with percentage {perc}") if print_level > 1 else None
    print(f"Unlabeled Size: {len(X_unlabeled)}, Labeled Size: {len(X_labeled)}") if print_level > 1 else None

    # Train the model on labeled data
    model.fit(X_labeled, y_labeled)

    if len(X_unlabeled) > 0:
        # Predict labels for the unlabeled data to create pseudo-labels
        pseudo_labels = model.predict(X_unlabeled)

        # Combine the labeled data and pseudo-labeled data
        X_combined = np.vstack((X_labeled, X_unlabeled))
        y_combined = np.concatenate((y_labeled, pseudo_labels))

        # Re-train the model on the combined dataset
        model.fit(X_combined, y_combined)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy of {accuracy}") if print_level > 1 else None
    return accuracy