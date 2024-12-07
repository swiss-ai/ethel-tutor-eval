import json
import matplotlib.pyplot as plt
import glob
import numpy as np

# Function to load and transform data from a JSON file
def load_evaluation_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)[0]  # Access the first element of the list
    return {
        'Math': data['mathematics'],
        'Physics': data['physics'],
        'Environmental Science': data['environmental science'],
        'Life Sciences': data['biology'],
        'Computer Science': data['chemistry']
    }

# Load all JSON files
json_files = glob.glob('/home/smaldo/Desktop/machine_learning_CS-433/ethel-tutor-eval/plots/mean_scores_*.json')
if not json_files:
    print("No JSON files found. Please check the directory and file pattern.")
else:
    mean_correctness_values = []
    model_names = []

    for i, file_path in enumerate(json_files, start=1):
        try:
            evaluation_data = load_evaluation_data(file_path)
            # Calculate mean correctness for each model
            mean_correctness = np.mean([evaluation_data[cat]['mean_correctness'] for cat in evaluation_data])
            mean_correctness_values.append(mean_correctness)
            model_names.append(f'Model {i}')
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if mean_correctness_values:
        # Define a list of colors
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'plum']

        # Plotting the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, mean_correctness_values, color=colors[:len(model_names)])
        plt.xlabel('Models')
        plt.ylabel('Mean Correctness')
        plt.title('Mean Correctness for Each Model')
        plt.ylim(0, 3)  # Assuming the scale is from 0 to 3
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No data to plot.")