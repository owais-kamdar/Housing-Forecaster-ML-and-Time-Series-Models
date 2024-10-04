import matplotlib.pyplot as plt

def plot_real_vs_predicted(y_test, y_pred, model_name, dates, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, y_test.reset_index(drop=True), label='Actual Values', color='blue', linewidth=2, marker='o')
    plt.plot(dates, y_pred, label=f'{model_name} Predictions', color='red', linestyle='--', marker='x')
    plt.title(f"Actual vs Predicted Values - {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Date (Month-Year)", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylabel("Values", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_model_comparisons(y_test, model_results, dates, model_colors, save_path=None):
    plt.figure(figsize=(10, 6))

    # Plot the actual values
    plt.plot(dates, y_test.reset_index(drop=True), label='Actual Values', color='blue', linewidth=2, marker='o')

    # Plot the predictions for each model
    for model_name, y_pred in model_results.items():
        color = model_colors.get(model_name, 'black')  # Assign color for each model
        plt.plot(dates, y_pred, label=f'{model_name} Predictions', color=color, linestyle='--', marker='x')

    plt.title("Comparison of Model Predictions", fontsize=14, fontweight='bold')
    plt.xlabel("Date (Month-Year)", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

