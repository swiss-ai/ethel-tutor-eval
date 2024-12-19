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
    plt.figure(figsize=(5, 3))

    for model_name, data in datasets:
        # Extract n_shots and accuracies from the data
        n_shots = [item['n_shots'] for item in data]
        accuracies = [item['accuracy'] for item in data]

        # Plot data for the current model
        plt.plot(n_shots, accuracies, marker='o', linestyle='-', label=model_name)

    # Add labels and title
    plt.xlabel('Number of Shots')
    plt.ylabel('Accuracy')
    plt.title('MGSM language accuracy Comparison')
    plt.xticks()  # Rotate x-axis labels for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
data1 = [ #llama-3.1 french
    {"n_shots": "0", "accuracy": 0.715},
    {"n_shots": "2", "accuracy": 0.824},
    {"n_shots": "4", "accuracy": 0.848},
    {"n_shots": "6", "accuracy": 0.863},
    {"n_shots": "8", "accuracy": 0.836},
]
data2 = [ #llama-3.1 german
    {"n_shots": "0", "accuracy": 0.592},
    {"n_shots": "2", "accuracy": 0.80},
    {"n_shots": "4", "accuracy": 0.811},
    {"n_shots": "6", "accuracy": 0.828},
    {"n_shots": "8", "accuracy": 0.828}
]

datasets = [
    ("Llama-3.1 french MGSM", data1),
    ("Llama-3.1 german MGSM", data2)
]

plot_model_accuracies(datasets)
