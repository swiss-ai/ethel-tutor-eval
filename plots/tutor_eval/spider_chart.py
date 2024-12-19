# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import glob
#
# # Function to load and transform data from a JSON file
# def load_evaluation_data(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     return {
#         'Math': data['mathematics'],
#         'Physics': data['physics'],
#         'Environmental Science': data['environmental science'],
#         'Life Sciences': data['biology'],
#         'Computer Science': data['chemistry']
#     }
#
# categories = ['Pre-Algebra', 'Algebra', 'Number Theory', 'Counting Probability', 'Geometry', 'Intermediate Algebra', 'Pre-Caculus']
#
# def create_radar_chart(data_list, categories, ax, title, value_type):
#     num_vars = len(categories)
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     angles += angles[:1]  # Complete the circle
#
#     ax.set_theta_offset(np.pi / 2)
#     ax.set_theta_direction(-1)
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(categories)
#
#     ax.set_rscale('linear')
#     ax.set_yticks([0, 1, 2, 3])
#     ax.set_yticklabels(["0", "1", "2", "3"])
#     ax.set_ylim(0, 3)
#
#     ax.spines['polar'].set_visible(False)
#
#
#     for i, data in enumerate(data_list):
#         values = [data[cat][value_type] for cat in categories]
#         values += values[:1]
#
#         ax.plot(angles, values, label=f"Model {i+1} {value_type.capitalize()}")
#         ax.fill(angles, values, alpha=0.1)
#
#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
#     ax.set_title(title)
#
# # Load all JSON files and create separate radar charts for correctness and presentation
# json_files = glob.glob('/plots/tutor_eval/mean_scores_2.json')
# if not json_files:
#     print("No JSON files found. Please check the directory and file pattern.")
# else:
#     all_data = []
#     for file_path in json_files:
#         try:
#             evaluation_data = load_evaluation_data(file_path)
#             all_data.append(evaluation_data)
#         except Exception as e:
#             print(f"Error processing file {file_path}: {e}")
#
#     if all_data:
#         fig, axs = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'polar': True})
#
#         create_radar_chart(all_data, categories, axs[0], 'Model Correctness Across Categories', 'mean_correctness')
#         create_radar_chart(all_data, categories, axs[1], 'Model Presentation Across Categories', 'mean_presentation')
#
#         plt.tight_layout()
#         plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Define the categories and data
categories = ['Pre-Algebra', 'Algebra', 'Number Theory',
              'Counting/Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']

# Hardcoded data for the models
ethel_70b_magpie = [0.783, 0.747, 0.513, 0.568, 0.432, 0.356, 0.361]
ethel_70b_tutorch = [0.404, 0.365, 0.178, 0.230, 0.175, 0.110, 0.130]

# Close the loop by adding the first value to the end for both models
categories += categories[:1]
ethel_70b_magpie += ethel_70b_magpie[:1]
ethel_70b_tutorch += ethel_70b_tutorch[:1]

# Calculate the angles for the radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True).tolist()

# Radar chart settings
fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'polar': True})

# Plot data for both models
ax.plot(angles, ethel_70b_magpie, label='ethel-70b-magpie', linewidth=2, color='tab:blue')
ax.fill(angles, ethel_70b_magpie, alpha=0.25, color='tab:blue')

ax.plot(angles, ethel_70b_tutorch, label='ethel-70b-tutorch', linewidth=2, color='tab:orange')
ax.fill(angles, ethel_70b_tutorch, alpha=0.25, color='tab:orange')

# Add labels with larger font size
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories[:-1], fontsize=12)  # Increase category label size
ax.tick_params(axis='y', labelsize=12)  # Increase y-axis label size

# Remove the bold black circle (spines)
ax.spines['polar'].set_visible(False)

# Add title with larger font
ax.set_title('Model Performance in Math Categories', size=14, pad=15)

# Add legend outside the plot to avoid overlapping text
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), fontsize=12, frameon=False)

# Show the chart
plt.tight_layout()
plt.show()
