import plotly.graph_objects as go
from partition import partition
from export import export
from train import train_and_evaluate
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.naive_bayes import GaussianNB

AVG_AMT = 1000
PERCENTAGES = [i for i in range(5, 101, 5)]
PERCENTAGES.insert(0, 1)
DATA_PATH = 'data\Dry_Bean_Dataset.arff'
WRITE_PATH = 'output'
SEMI_SUPERVISED_MODELS = [SVC(), RandomForestClassifier(), LabelPropagation(), LabelSpreading(), GaussianNB()]
SUPERVISED_MODELS = [SVC(), MLPClassifier(), RandomForestClassifier(), LabelPropagation(), LabelSpreading()]
PRINT_LEVEL = 2

def run(semi):
    models = SEMI_SUPERVISED_MODELS if semi else SUPERVISED_MODELS
    for model in models:
        all_data = []

        # Create a partial function that includes the current model
        name = type(model).__name__
        training_function = lambda X_l, y_l, X_u, X_t, y_t, i, p, n, pr: train_and_evaluate(model, X_l, y_l, X_u, X_t, y_t, i, p, n, pr)

        percentages, avg_accuracies = partition(training_function=training_function, is_semi_supervised=semi, 
                                                avg_amt=AVG_AMT, path=DATA_PATH, percentages=PERCENTAGES, name=name, print_level=PRINT_LEVEL)
        all_data.append(go.Scatter(x=percentages, y=avg_accuracies, mode='lines+markers', name=f'{name} Avg {AVG_AMT}'))
        
        export(data=all_data, percentages=percentages, avg_accuracies=avg_accuracies, 
               path=WRITE_PATH, name=name, avg_amt=AVG_AMT, semi=semi)

if __name__ == "__main__":
    run(semi=False)
    run(semi=True)