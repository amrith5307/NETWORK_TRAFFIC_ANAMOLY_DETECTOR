import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

def plot_final_comparison(summary_list):
    """Generates a bar chart comparing Accuracy, Precision, Recall, and F1."""
    df = pd.DataFrame(summary_list)
    df_melted = df.melt(id_vars='MODEL', var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='MODEL', y='Score', hue='Metric')
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/final_metrics_comparison.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Generates a heatmap showing true positives vs false positives."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('AI Prediction')
    plt.tight_layout()

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/{model_name}_cm.png")
    plt.close()