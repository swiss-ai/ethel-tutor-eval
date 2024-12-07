import matplotlib.pyplot as plt
import numpy as np

# Example data
categories = ['Math', 'Physics', 'Environmental Science', 'Life Sciences', 'Computer Science']
models = ['GPT-4', 'GPT-3.5', 'Llemma-34B-MathMix', 'Mistral-7B-V2', 'Zephyr-7B']

# Radar chart data
performance = {
    'GPT-4': [95, 90, 85, 80, 88],
    'GPT-3.5': [85, 83, 80, 75, 78],
    'Llemma-34B-MathMix': [78, 76, 74, 70, 72],
    'Mistral-7B-V2': [75, 72, 68, 65, 70],
    'Zephyr-7B': [65, 60, 58, 55, 62]
}

# Horizontal bar chart data
ranked_scores = [np.mean(performance[model]) for model in models]

# Create radar chart
def create_radar_chart(data, categories, ax, title):
    num_vars = len(categories)

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Radar chart
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axis per variable and add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Draw ylabels
    ax.set_rscale('linear')
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.set_ylim(0, 100)

    # Plot each model
    for model, values in data.items():
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title(title)

# Create horizontal bar chart
def create_bar_chart(scores, labels, ax, title):
    ax.barh(labels, scores, color='skyblue')
    ax.set_xlabel('Average Performance')
    ax.set_title(title)
    ax.invert_yaxis()

# Plot both charts
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'polar': True})

fig, ax1 = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})


# Radar chart
create_radar_chart(performance, categories, ax1, 'Model Performance Across Categories')

# Bar chart
#create_bar_chart(ranked_scores, models, ax2, 'Model Rankings')

plt.tight_layout()
plt.show()