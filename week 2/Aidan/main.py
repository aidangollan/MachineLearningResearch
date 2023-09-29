from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.io import arff
from joblib import Parallel, delayed
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import plotly.io as pio

def train_and_evaluate(X_sample, y_sample, X_test, y_test, i):
    print(f"running test iteration {i}")
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), alpha=0.0001, solver='adam', max_iter=10000)
    clf.fit(X_sample, y_sample)
    y_pred = clf.predict(X_test)
    print(f"test iteration {i} complete at time {datetime.now()}")
    return accuracy_score(y_test, y_pred)

def main_optimized(avg_per_perc):
    # Load the data
    data, meta = arff.loadarff('C:\\Users\\Aidan\\MachineLearningResearch\\data\\Dry_Bean_Dataset.arff')  # Adjust path for your setup
    df = pd.DataFrame(data)
    
    # Convert byte strings to regular strings for the target variable
    df['Class'] = df['Class'].str.decode('utf-8')
    
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
        sample_size = int((perc / 100) * len(X_train))
        
        if perc == 100:
            X_sample, y_sample = X_train, y_train
        else:
            X_sample, _, y_sample, _ = train_test_split(X_train, y_train, train_size=sample_size, shuffle=True)
        
        print(f"running percentage {perc} with avg amt {avg_per_perc} at time {datetime.now()}")
        accuracies = Parallel(n_jobs=3)(delayed(train_and_evaluate)(
            X_sample, y_sample, X_test, y_test, i) for i in range(avg_per_perc))
        
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
    pio.write_image(fig, 'plot.png')
