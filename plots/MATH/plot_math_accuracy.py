import matplotlib.pyplot as plt

def plot_model_accuracies(datasets):
    """
    Plots accuracies for multiple models.

    Parameters:
        datasets (list of tuples): A list of tuples where each tuple contains:
            - model_name (str): The name of the model.
            - data (list of dict): A list of dictionaries where each dictionary contains:
                - 'n_shots' (str): The number of shots.
                - 'accuracy' (float): The accuracy of the model.
    """
    plt.figure(figsize=(10, 6))

    for model_name, data in datasets:
        # Extract n_shots and accuracies from the data
        n_shots = [item['n_shots'] for item in data]
        accuracies = [item['accuracy'] for item in data]

        # Plot data for the current model
        plt.plot(n_shots, accuracies, marker='o', linestyle='-', label=model_name)

    # Add labels and title
    plt.xlabel('Number of Shots')
    plt.ylabel('Accuracy')
    plt.title('MATH dataset Comparison')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
data1 = [
    {"n_shots": "0", "accuracy": 0.85},
    {"n_shots": "2", "accuracy": 0.38},
    {"n_shots": "4", "accuracy": 0.22},
    {"n_shots": "8", "accuracy": 0.58}
]
data2 = [
    {"n_shots": "0", "accuracy": 0.15},
    {"n_shots": "2", "accuracy": 0.68},
    {"n_shots": "4", "accuracy": 0.02},
    {"n_shots": "8", "accuracy": 0.88}
]
data3 = [
    {"n_shots": "0", "accuracy": 0.35},
    {"n_shots": "2", "accuracy": 0.78},
    {"n_shots": "4", "accuracy": 0.92},
    {"n_shots": "8", "accuracy": 0.88}
]

datasets = [
    ("Model A", data1),
    ("Model B", data2),
    ("Model C", data3)
]

plot_model_accuracies(datasets)
