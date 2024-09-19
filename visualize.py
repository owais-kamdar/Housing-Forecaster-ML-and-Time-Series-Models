import matplotlib.pyplot as plt

def plot_real_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, label='Actual Values', color='blue')
    plt.plot(range(len(y_pred)), y_pred, label=f'{model_name} Predicted Values', color='red')
    plt.title(f"Real vs Predicted Values - {model_name}")
    plt.xlabel("Data Points")
    plt.ylabel("Values")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_model_comparisons(model_results, y_test):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Values', color='blue')
    for model_name, y_pred in model_results.items():
        plt.plot(y_pred, label=f'{model_name} Predictions')
    plt.title("Comparison of Model Predictions")
    plt.xlabel("Data Points")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.tight_layout()
    plt.show()
