
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
ax.plot(angles, ethel_70b_magpie, label='Llama-3.1-70B', linewidth=2, color='tab:blue')
ax.fill(angles, ethel_70b_magpie, alpha=0.25, color='tab:blue')

ax.plot(angles, ethel_70b_tutorch, label='Llama-3.2', linewidth=2, color='tab:orange')
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
