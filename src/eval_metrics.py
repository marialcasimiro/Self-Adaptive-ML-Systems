#!/usr/bin/env python

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (  # auc,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    zero_one_loss,
)


def compute_and_plot_roc_curve(labels: list, scores: list):
    """compute roc curve, fpr, tpr, and threshold values"""

    if (
        isinstance(scores, list)
        and isinstance(labels, list)
        and isinstance(scores[0], np.ndarray)
        and isinstance(labels[0], np.ndarray)
    ):
        scores_transformed = []
        real_labels_transformed = []

        for score in scores:
            scores_transformed.extend(score)

        for label in labels:
            real_labels_transformed.extend(label)
    else:
        scores_transformed = scores
        real_labels_transformed = labels

    # Compute Receiver operating characteristic (ROC)
    fpr, tpr, thresholds = roc_curve(real_labels_transformed, scores_transformed)

    # Compute Area Under the Curve (AUC) using the trapezoidal rule
    # roc_auc = auc(fpr, tpr)

    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

    return fpr, tpr, thresholds


def get_predictions(threshold: float, scores: list):
    """compute predictions based on the selected threshold"""

    predictions = []

    if isinstance(scores, np.ndarray):
        for score in scores:
            if score <= threshold:
                predictions.append(0.0)
            else:
                predictions.append(1.0)
    else:
        for batch in scores:
            preds = []
            for score in batch:
                if score <= threshold:
                    preds.append(0.0)
                else:
                    preds.append(1.0)
            predictions.append(preds)

    return np.array(predictions)


def get_threshold_at_fpr(target_fpr: float, fpr: np.array, thresholds: np.array):
    """compute threshold for target FPR"""

    # 1. find index where FPR is closer to the desired value
    idx = np.where(fpr == fpr[fpr < target_fpr].max())[0]
    idx = idx[0]

    # 2. find threshold that leads to that FPR
    threshold = thresholds[idx]

    if isinstance(threshold, np.ndarray):
        return threshold[0]
    else:
        return threshold


def compute_metrics(labels: list, predictions: list, scores: list, max_fpr: float):
    """compute metrics for a specific FPR (e.g. 10%)"""
    print("\n[D] Computing test metrics.")
    roc_auc = []
    acc = []
    _precision_score = []
    _recall_score = []
    _f1_score = []
    loss = []

    for idx, label in enumerate(labels):
        # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
        roc_auc.append(roc_auc_score(label, scores[idx], max_fpr=max_fpr))

        # Accuracy classification score
        acc.append(accuracy_score(label, predictions[idx]))

        # Compute the precision
        _precision_score.append(precision_score(label, predictions[idx]))

        # Compute the recall
        _recall_score.append(recall_score(label, predictions[idx]))

        # Compute the F1 score, also known as balanced F-score or F-measure
        _f1_score.append(f1_score(label, predictions[idx]))

        # Return the fraction of misclassifications
        loss.append(zero_one_loss(label, predictions[idx]))

    metrics_df = pd.DataFrame(
        {
            "real_labels": labels,
            "scores": scores,
            "predictions": predictions,
            "roc_auc": roc_auc,
            "acc": acc,
            "precisionScore": _precision_score,
            "recallScore": _recall_score,
            "f1Score": _f1_score,
            "loss": loss,
        }
    )

    return metrics_df, roc_auc, acc, _precision_score, _recall_score, _f1_score, loss
