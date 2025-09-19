import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return {'accuracy': acc, 'confusion_matrix': cm, 'report': report}
