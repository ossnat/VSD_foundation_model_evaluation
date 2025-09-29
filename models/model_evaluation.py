### Single model evaluation
Usage:
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