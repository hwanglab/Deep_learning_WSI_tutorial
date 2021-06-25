import numpy as np
from sklearn.metrics import roc_curve


def find_optimal_threshold(y_true, y_score,
                           pos_label=None, sample_weight=None, drop_intermediate=True):
    """
    Uses sklearn.metrics.roc_curve to find the optimal threshold using the point on curve closest to (0,1) method

    Returns: Optimal threshold, sensitivity, specificity
    """
    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
        pos_label=pos_label,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate
    )
    sensitivity = tpr
    specificity = 1 - fpr
    # Distance from the upper-left corner
    distance = np.sqrt((1 - sensitivity)**2 + (1 - specificity)**2)
    best_idx = np.argmin(distance)

    return thresholds[best_idx], sensitivity[best_idx], specificity[best_idx]
