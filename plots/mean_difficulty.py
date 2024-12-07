# import json
# import numpy as np
#
# # Load the JSON data
# with open('/home/smaldo/Desktop/machine_learning_CS-433/ethel-tutor-eval/plots/model_val_1.json', 'r') as f:
#     data = json.load(f)
#
#
# # Initialize dictionaries to store sums and counts
# difficulty_sums = {
#     'easy': {'correctness': 0, 'presentation': 0, 'count': 0},
#     'medium': {'correctness': 0, 'presentation': 0, 'count': 0},
#     'hard': {'correctness': 0, 'presentation': 0, 'count': 0}
# }
#
# # Process each entry in the data
# for entry in data:
#     difficulty = entry['difficulty']
#     difficulty_sums[difficulty]['correctness'] += int(entry['correctness'])
#     difficulty_sums[difficulty]['presentation'] += int(entry['presentation'])
#     difficulty_sums[difficulty]['count'] += 1
#
# # Calculate means
# difficulty_means = {}
# for difficulty, values in difficulty_sums.items():
#     if values['count'] > 0:
#         difficulty_means[difficulty] = {
#             'mean_correctness': values['correctness'] / values['count'],
#             'mean_presentation': values['presentation'] / values['count']
#         }
#     else:
#         difficulty_means[difficulty] = {
#             'mean_correctness': 0,
#             'mean_presentation': 0
#         }
#
# # Save the results to a JSON file
# with open('/home/smaldo/Desktop/machine_learning_CS-433/ethel-tutor-eval/plots/mean_difficulty_1.json', 'w') as f:
#     json.dump(difficulty_means, f, indent=4)
#
# # Print the results
# for difficulty, means in difficulty_means.items():
#     print(f"{difficulty.capitalize()} - Mean Correctness: {means['mean_correctness']}, Mean Presentation: {means['mean_presentation']}")

import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
file_path = '/home/smaldo/Desktop/machine_learning_CS-433/ethel-tutor-eval/plots/mean_difficulty_1.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract data for plotting
difficulties = list(data.keys())
mean_correctness_values = [data[difficulty]['mean_correctness'] for difficulty in difficulties]
mean_presentation_values = [data[difficulty]['mean_presentation'] for difficulty in difficulties]

# Define a list of colors
colors = ['skyblue', 'lightgreen', 'salmon']

# Plotting the bar chart
x = np.arange(len(difficulties))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, mean_correctness_values, width, label='Correctness', color=colors[0])
rects2 = ax.bar(x + width/2, mean_presentation_values, width, label='Presentation', color=colors[1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Difficulty')
ax.set_ylabel('Scores')
ax.set_title('Mean Correctness and Presentation by Difficulty')
ax.set_xticks(x)
ax.set_xticklabels(difficulties)
ax.legend()

fig.tight_layout()

plt.show()