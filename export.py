import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import os

def export(data, percentages, avg_accuracies, path, name, avg_amt, semi):
    type = "SemiSupervised" if semi else "Supervised"
    fig = go.Figure(data=data)
    fig.update_layout(title='Average Accuracy vs. Percentage of Training Data Used',
                      xaxis_title='Percentage of Training Data Used',
                      yaxis_title='Average Accuracy')

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({'Percentage': percentages, 'Average Accuracy': avg_accuracies})
    
    # Define the base file name
    base_file_name = f'{name}{type}{avg_amt}AVG'
    
    # Construct full file paths
    csv_file_path = os.path.join(path, f'{base_file_name}.csv')
    png_file_path = os.path.join(path, f'{base_file_name}.png')

    # Write the results to a CSV file
    results_df.to_csv(csv_file_path, index=False)
    
    # Save the figure to a PNG file
    pio.write_image(fig, png_file_path)