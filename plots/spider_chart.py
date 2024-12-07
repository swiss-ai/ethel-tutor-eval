import json
import matplotlib.pyplot as plt
import numpy as np
import glob

# Function to load and transform data from a JSON file
def load_evaluation_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {
        'Math': data['mathematics'],
        'Physics': data['physics'],
        'Environmental Science': data['environmental science'],
        'Life Sciences': data['biology'],
        'Computer Science': data['chemistry']
    }

categories = ['Math', 'Physics', 'Environmental Science', 'Life Sciences', 'Computer Science']

def create_radar_chart(data_list, categories, ax, title, value_type):
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    ax.set_rscale('linear')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0", "1", "2", "3"])
    ax.set_ylim(0, 3)

    ax.spines['polar'].set_visible(False)


    for i, data in enumerate(data_list):
        values = [data[cat][value_type] for cat in categories]
        values += values[:1]

        ax.plot(angles, values, label=f"Model {i+1} {value_type.capitalize()}")
        ax.fill(angles, values, alpha=0.1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title(title)

# Load all JSON files and create separate radar charts for correctness and presentation
json_files = glob.glob('/home/smaldo/Desktop/machine_learning_CS-433/ethel-tutor-eval/plots/mean_scores_*.json')
if not json_files:
    print("No JSON files found. Please check the directory and file pattern.")
else:
    all_data = []
    for file_path in json_files:
        try:
            evaluation_data = load_evaluation_data(file_path)
            all_data.append(evaluation_data)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if all_data:
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'polar': True})
        
        create_radar_chart(all_data, categories, axs[0], 'Model Correctness Across Categories', 'mean_correctness')
        create_radar_chart(all_data, categories, axs[1], 'Model Presentation Across Categories', 'mean_presentation')
        
        plt.tight_layout()
        plt.show()