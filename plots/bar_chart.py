# import json
# import matplotlib.pyplot as plt
# import glob
# import numpy as np
#
# # Function to load and transform data from a JSON file
# def load_evaluation_data(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)[0]  # Access the first element of the list
#     return {
#         'Math': data['mathematics'],
#         'Physics': data['physics'],
#         'Environmental Science': data['environmental science'],
#         'Life Sciences': data['biology'],
#         'Computer Science': data['chemistry']
#     }
#
# # Load all JSON files
# json_files = glob.glob('/home/smaldo/Desktop/machine_learning_CS-433/ethel-tutor-eval/plots/mean_difficulty_*.json')
# if not json_files:
#     print("No JSON files found. Please check the directory and file pattern.")
# else:
#     mean_correctness_values = []
#     model_names = []
#
#     for i, file_path in enumerate(json_files, start=1):
#         try:
#             evaluation_data = load_evaluation_data(file_path)
#             # Calculate mean correctness for each model
#             mean_correctness = np.mean([evaluation_data[cat]['mean_correctness'] for cat in evaluation_data])
#             mean_correctness_values.append(mean_correctness)
#             model_names.append(f'Model {i}')
#         except Exception as e:
#             print(f"Error processing file {file_path}: {e}")
#
#     if mean_correctness_values:
#         # Define a list of colors
#         colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'plum']
#
#         # Plotting the bar chart
#         plt.figure(figsize=(10, 6))
#         plt.bar(model_names, mean_correctness_values, color=colors[:len(model_names)])
#         plt.xlabel('Models')
#         plt.ylabel('Mean Correctness')
#         plt.title('Mean Correctness for Each Model')
#         plt.ylim(0, 3)  # Assuming the scale is from 0 to 3
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No data to plot.")

import json
import matplotlib.pyplot as plt
import numpy as np
import glob

# Load all JSON files
json_files = glob.glob('/home/smaldo/Desktop/machine_learning_CS-433/ethel-tutor-eval/plots/mean_difficulty_*.json')

if not json_files:
    print("No JSON files found. Please check the directory and file pattern.")
else:
    correctness_data = {}

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            model_name = file_path.split('/')[-1].replace('.json', '')
            correctness_data[model_name] = [data[difficulty]['mean_correctness'] for difficulty in data]

    # Plotting the bar chart for correctness
    difficulties = list(data.keys())
    x = np.arange(len(difficulties))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (model_name, correctness_values) in enumerate(correctness_data.items()):
        ax.bar(x + i * width, correctness_values, width, label=model_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Difficulty')
    ax.set_ylabel('Mean Correctness')
    ax.set_title('Mean Correctness by Difficulty for Each Model')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(difficulties)
    ax.legend()

    fig.tight_layout()
    plt.show()