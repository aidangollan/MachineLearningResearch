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

def train_and_evaluate(X_labeled, y_labeled, X_unlabeled, X_test, y_test, i, perc):
    print(f"running test iteration {i} with percentage {perc}")
    
    # Train the Random Forest classifier on the labeled data
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_labeled, y_labeled)

    # Check if there are unlabeled data
    if X_unlabeled.size > 0:
        # Predict labels for the unlabeled data
        pseudo_labels = rf_classifier.predict(X_unlabeled)

        # Combine the labeled data and pseudo-labeled data
        X_combined = np.vstack((X_labeled, X_unlabeled))
        y_combined = np.concatenate((y_labeled, pseudo_labels))

        # Re-train the Random Forest classifier on the combined dataset
        rf_classifier.fit(X_combined, y_combined)
    else:
        # No unlabeled data, so we use only the labeled data
        rf_classifier.fit(X_labeled, y_labeled)

    # Evaluate the re-trained model on the test set
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy

def main_optimized(avg_per_perc):
    # Load the data
    data, meta = arff.loadarff('C:\\Users\\Aidan\\MachineLearningResearch\\data\\Dry_Bean_Dataset.arff')
    df = pd.DataFrame(data)
    
    # Convert byte strings to regular strings for the target variable
    df['Class'] = df['Class'].str.decode('utf-8')
    
    # Encode the labels to integers
    label_encoder = LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['Class'])
    
    # Split the features and target variable
    X = df.drop(columns='Class')
    y = df['Class']
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Splitting the dataset into training and testing sets (50% each)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    percentages = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    avg_accuracies = []
    
    print(f"starting training for average amt {avg_per_perc} at time {datetime.now()}")

    for perc in percentages:
        time_now = datetime.now()
        labeled_size = int((perc / 100) * len(X_train))
        
        if perc == 100:  # If 100%, don't split again. Use all data as labeled.
            X_labeled, y_labeled = X_train, y_train
            X_unlabeled = np.array([])  # Empty array for consistency
        else:
            X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X_train, y_train, train_size=labeled_size, shuffle=True)
        
        print(f"running percentage {perc} with avg amt {avg_per_perc} at time {datetime.now()}")
        accuracies = Parallel(n_jobs=3)(delayed(train_and_evaluate)(
            X_labeled, y_labeled, X_unlabeled, X_test, y_test, i, perc) for i in range(avg_per_perc))
        
        time_now = datetime.now() - time_now
        print(f"percentage {perc} with avg amt {avg_per_perc} completed at time: {datetime.now()}, total time: {time_now}")
        avg_accuracies.append(np.mean(accuracies))
    
    return percentages, avg_accuracies

if __name__ == "__main__":
    all_data = []
    avgs = [1000]
    for i in avgs:
        percentages, avg_accuracies = main_optimized(i)
        all_data.append(go.Scatter(x=percentages, y=avg_accuracies, mode='lines+markers', name=f'Avg {i}'))

    fig = go.Figure(data=all_data)
    fig.update_layout(title='Average Accuracy vs. Percentage of Training Data Used',
                      xaxis_title='Percentage of Training Data Used',
                      yaxis_title='Average Accuracy')

    # create a DataFrame to store the results
    results_df = pd.DataFrame({'Percentage': percentages, 'Average Accuracy': avg_accuracies})
    
    # define the base file name
    base_file_name = 'LabelPropLabeledAndUnlabledData1000AVG'
    
    # write the results to a csv file
    results_df.to_csv(f'{base_file_name}.csv', index=False)
    
    # save the figure to a png file
    pio.write_image(fig, f'{base_file_name}.png')

