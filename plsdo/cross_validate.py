"""Cross-validation for discriminatory PLS."""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


def run_cv(
    X: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 100,
    n_components: int = 3,
    seed: int = 42,
) -> dict:
    """Run Repeated Stratified K-Fold CV for discriminatory PLS.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_features)
        Continuous data matrix (MRI, behaviour, etc.).
    labels : ndarray, shape (n_subjects,)
        Integer group labels.
    n_splits : int
        Number of CV folds.
    n_repeats : int
        Number of CV repeats.
    n_components : int
        Number of PLS components.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        mean_accuracy, mean_balanced_accuracy, fold_results (DataFrame),
        true_labels, pred_labels, confusion_matrix
    """
    n_groups = len(np.unique(labels))
    Y_dummy = np.eye(n_groups)[labels]

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    fold_results = []
    all_true = []
    all_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, labels)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y_dummy[train_idx]
        labels_test = labels[test_idx]

        # Standardise: fit on train, apply to both
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        # Fit PLS and predict
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_train_std, Y_train)
        Y_pred = pls.predict(X_test_std)
        pred_labels = np.argmax(Y_pred, axis=1)

        acc = accuracy_score(labels_test, pred_labels)
        bal_acc = balanced_accuracy_score(labels_test, pred_labels)

        fold_results.append({
            "fold": fold_idx,
            "repeat": fold_idx // n_splits,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "n_test": len(test_idx),
        })

        all_true.extend(labels_test)
        all_pred.extend(pred_labels)

    true_all = np.array(all_true)
    pred_all = np.array(all_pred)
    fold_df = pd.DataFrame(fold_results)

    return {
        "mean_accuracy": fold_df["accuracy"].mean(),
        "mean_balanced_accuracy": fold_df["balanced_accuracy"].mean(),
        "fold_results": fold_df,
        "true_labels": true_all,
        "pred_labels": pred_all,
        "confusion_matrix": confusion_matrix(
            true_all, pred_all, normalize="true"
        ),
    }


def permutation_test_cv(
    X: np.ndarray,
    labels: np.ndarray,
    observed_accuracy: float,
    n_splits: int = 5,
    n_repeats: int = 1,
    n_components: int = 3,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test for CV accuracy significance.

    Shuffles group labels and repeats CV to build a null distribution.

    Parameters
    ----------
    X : ndarray
        Continuous data matrix.
    labels : ndarray
        Integer group labels.
    observed_accuracy : float
        The observed CV accuracy to test against.
    n_splits, n_repeats, n_components : int
        CV parameters (use fewer repeats per permutation for speed).
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: p_value, null_accuracies
    """
    rng = np.random.RandomState(seed)
    null_accs = []

    for perm_i in range(n_permutations):
        shuffled = rng.permutation(labels)
        result = run_cv(
            X, shuffled, n_splits=n_splits, n_repeats=n_repeats,
            n_components=n_components, seed=perm_i,
        )
        null_accs.append(result["mean_accuracy"])

    null_accs = np.array(null_accs)
    p_value = np.mean(null_accs >= observed_accuracy)

    return {
        "p_value": p_value,
        "null_accuracies": null_accs,
    }
