import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, silhouette_samples
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize


def alpha_diversity(abundance):
    values = abundance.to_numpy(dtype=float)
    row_sum = values.sum(axis=1)
    relative = np.divide(values, row_sum[:, None], out=np.zeros_like(values), where=row_sum[:, None] > 0)
    log_relative = np.where(relative > 0, np.log(relative), 0.0)

    shannon = -np.sum(relative * log_relative, axis=1)
    richness = np.sum(values > 0, axis=1).astype(float)
    evenness = np.divide(shannon, np.log(richness), out=np.zeros_like(shannon), where=richness > 1)

    rounded = np.rint(values).astype(int)
    f1 = np.sum(rounded == 1, axis=1).astype(float)
    f2 = np.sum(rounded == 2, axis=1).astype(float)
    chao1 = richness + np.where(f2 > 0, (f1 * f1) / (2.0 * f2), (f1 * (f1 - 1.0)) / 2.0)

    return pd.DataFrame(
        {
            "Shannon": shannon,
            "Richness": richness,
            "Chao1": chao1,
            "Evenness": evenness,
        },
        index=abundance.index,
    )


def batch_r2_from_distance(distance_matrix, labels):
    ids = list(distance_matrix.ids)
    labels = labels.loc[ids].astype(str).to_numpy()
    distances = np.asarray(distance_matrix.data, dtype=float)
    squared_distances = distances ** 2
    n = squared_distances.shape[0]

    if n <= 1:
        return np.nan

    total_sum = squared_distances.sum() / (2.0 * n)
    within_sum = 0.0

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) <= 1:
            continue
        within_sum += squared_distances[np.ix_(indices, indices)].sum() / (2.0 * len(indices))

    if total_sum <= 0:
        return np.nan

    return float(max(0.0, (total_sum - within_sum) / total_sum))


def batch_silhouette_values(distance_matrix, labels):
    ids = list(distance_matrix.ids)
    labels = labels.loc[ids].astype(str).to_numpy()
    distances = np.asarray(distance_matrix.data, dtype=float)

    if len(np.unique(labels)) < 2:
        return np.full(len(labels), np.nan)

    return silhouette_samples(distances, labels, metric="precomputed")


def leave_one_group_auc(features, labels, groups, n_estimators=100, random_state=42):
    aucs = {}

    for group in groups.unique():
        test_mask = groups == group
        train_mask = ~test_mask
        y_test = labels.loc[test_mask]

        if y_test.nunique() < 2:
            continue

        classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        classifier.fit(features.loc[train_mask], labels.loc[train_mask])
        probabilities = classifier.predict_proba(features.loc[test_mask])[:, 1]
        aucs[str(group)] = roc_auc_score(y_test, probabilities)

    return pd.Series(aucs, name="auc")


def cross_validated_scores(features, labels, classifier, n_splits=3, random_state=42):
    labels = np.asarray(labels)
    predictions = np.empty_like(labels)
    classes = np.unique(labels)
    scores = np.empty((len(labels), len(classes)), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, test_index in splitter.split(features, labels):
        model = classifier()
        model.fit(features[train_index], labels[train_index])
        predictions[test_index] = model.predict(features[test_index])
        decision = model.decision_function(features[test_index])

        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T

        scores[test_index] = decision

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "score": scores,
    }


def roc_tables(labels, scores, class_names, representation):
    classes = np.arange(len(class_names))
    binarized = label_binarize(labels, classes=classes)
    curve_records = []
    auc_records = []
    mean_fpr = np.linspace(0.0, 1.0, 201)
    interpolated_tprs = []

    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(binarized[:, class_index], scores[:, class_index])
        class_auc = auc(fpr, tpr)
        auc_records.append({"representation": representation, "class": class_name, "auc": float(class_auc)})
        curve_records.extend(
            [
                {
                    "representation": representation,
                    "class": class_name,
                    "curve": "class",
                    "fpr": float(x),
                    "tpr": float(y),
                    "auc": float(class_auc),
                }
                for x, y in zip(fpr, tpr)
            ]
        )
        interpolated = np.interp(mean_fpr, fpr, tpr)
        interpolated[0] = 0.0
        interpolated_tprs.append(interpolated)

    macro_tpr = np.mean(np.vstack(interpolated_tprs), axis=0)
    macro_tpr[-1] = 1.0
    macro_auc = auc(mean_fpr, macro_tpr)
    auc_records.append({"representation": representation, "class": "macro", "auc": float(macro_auc)})
    curve_records.extend(
        [
            {
                "representation": representation,
                "class": "macro",
                "curve": "macro",
                "fpr": float(x),
                "tpr": float(y),
                "auc": float(macro_auc),
            }
            for x, y in zip(mean_fpr, macro_tpr)
        ]
    )

    return pd.DataFrame(curve_records), pd.DataFrame(auc_records)


def paired_or_unpaired_p(before, after, paired=True):
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]

    if len(before) == 0 or len(after) == 0:
        return np.nan

    try:
        if paired and len(before) == len(after):
            if np.allclose(before, after):
                return 1.0
            return float(wilcoxon(before, after, zero_method="wilcox").pvalue)
        return float(mannwhitneyu(before, after, alternative="two-sided").pvalue)
    except Exception:
        return np.nan
