## Batch evaluation:

import os
import json
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
# Assuming VideoClassifier and build_pooling are available from imported modules
from models.classifier import VideoClassifier # Ensure this import is present

def evaluate_and_save_model_results(model, test_loader, device, backbone_name, temporal_pooling_method, num_epochs, results_dir):
    """Evaluates a model and saves the results."""
    print(f"\nEvaluating model: Backbone={backbone_name}, Pooling={temporal_pooling_method}, Epochs={num_epochs}")

    # Perform evaluation
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for clips, labels in tqdm(test_loader, desc=f"Evaluating {backbone_name}-{temporal_pooling_method}-Epochs{num_epochs}"):
            B, T, C, H, W = clips.shape
            clips, labels = clips.to(device), labels.to(device)

            clips = clips.view(B*T, C, H, W)
            if clips.shape[1] == 1:
                 clips = clips.repeat(1, 3, 1, 1)

            logits = model(clips, B, T)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Create subdirectory for results including epochs
    model_results_subdir = os.path.join(results_dir, f"{backbone_name}_{temporal_pooling_method}_epochs{num_epochs}")
    os.makedirs(model_results_subdir, exist_ok=True)

    # Save confusion matrix plot
    fig, ax = plt.subplots()
    cmd_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Vertical", "Horizontal"])
    cmd_display.plot(ax=ax)
    plt.title(f"Confusion Matrix: {backbone_name}-{temporal_pooling_method}-Epochs{num_epochs}")
    plt.savefig(os.path.join(model_results_subdir, "confusion_matrix.png"))
    plt.close(fig) # Close the figure to free memory

    # Save numerical results
    results_data = {
        "backbone": backbone_name,
        "temporal_pooling": temporal_pooling_method,
        "num_epochs": num_epochs,
        "accuracy": accuracy,
        "auc": auc,
        "confusion_matrix": cm.tolist() # Convert numpy array to list for JSON serialization
    }
    results_filepath = os.path.join(model_results_subdir, "results.json")
    with open(results_filepath, "w") as f:
        json.dump(results_data, f, indent=4)

    print(f"Results saved to {model_results_subdir}")
    return results_data # Return results for summary report


def generate_summary_report(results_dir):
    """Generates a summary report from saved results."""
    print("\nGenerating summary report...")
    summary_list = []

    # Iterate through subdirectories in test_results_dir
    for subdir_name in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir_name)
        # Ensure it's a directory and contains results.json
        if os.path.isdir(subdir_path):
            results_filepath = os.path.join(subdir_path, "results.json")
            if os.path.exists(results_filepath):
                with open(results_filepath, "r") as f:
                    results_data = json.load(f)
                    summary_list.append(results_data)

    if not summary_list:
        print("No results found to generate a report.")
        return

    # Create a pandas DataFrame for easier reporting
    summary_df = pd.DataFrame(summary_list)
    # Reorder columns for the summary report
    summary_df = summary_df[['backbone', 'temporal_pooling', 'num_epochs', 'accuracy', 'auc']]
    # Sort for better readability (optional)
    summary_df = summary_df.sort_values(by=['backbone', 'temporal_pooling', 'num_epochs']).reset_index(drop=True)


    print("\n--- Model Performance Summary ---")
    print(summary_df.to_markdown(index=False)) # Use to_markdown for nice output in Colab

    # Optional: Save summary to a file
    summary_filepath = os.path.join(results_dir, "summary_report.md")
    with open(summary_filepath, "w") as f:
        f.write("### Model Performance Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
    print(f"\nSummary report saved to {summary_filepath}")

# Usage:

#############################

### Single model evaluation
# Usage:
# Train the model and get the history, trained model, and device
# test_accuracy, test_auc, test_cm = evaluate_model(trained_model, test_loader, device)

# print("\nEvaluation Results:")
# print(f"Test Accuracy: {test_accuracy:.4f}")
# print(f"Test AUC: {test_auc:.4f}")
# print("Confusion Matrix:")
# print(test_cm)

###
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm # Import tqdm for progress bar
import numpy as np # Import numpy

def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for clips, labels in tqdm(test_loader, desc="Evaluating"):
            B, T, C, H, W = clips.shape
            clips, labels = clips.to(device), labels.to(device)

            # Flatten (B,T,C,H,W) → (B*T,C,H,W)
            clips = clips.view(B*T, C, H, W)
            # Ensure clips have 3 channels if the model expects RGB (like ResNet/DINO)
            # This needs to be consistent with how the model was trained
            # Assuming the model expects 3 channels based on earlier preprocessing
            if clips.shape[1] == 1:
                 clips = clips.repeat(1, 3, 1, 1)


            logits = model(clips, B, T)
            probs = torch.softmax(logits, dim=1)[:, 1] # Get probability for the positive class

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print('All_labels: ', set(all_labels))
    print('All_preds: ', set(all_preds))

    # Plot confusion matrix
    fig, ax = plt.subplots() # Get figure and axes
    cmd_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Vertical", "Horizontal"])
    cmd_display.plot(ax=ax) # Pass axes to plot method
    plt.title("Confusion Matrix")
    # Remove explicit tick setting, rely on cmd_display.plot(ax=ax)
    # ax.set_xticks(np.arange(len(["Vertical", "Horizontal"])))
    # ax.set_yticks(np.arange(len(["Vertical", "Horizontal"])))
    # ax.set_xticklabels(["Vertical", "Horizontal"])
    # ax.set_yticklabels(["Vertical", "Horizontal"])
    plt.show()

    return accuracy, auc, cm