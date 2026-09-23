"""Official BGRL linear evaluation with sklearn compatibility shims.

Source: nerdslab/bgrl `bgrl/logistic_regression_eval.py`.

The Half-Hop paper (Appendix D) specifies L2-regularized logistic regression
with the liblinear solver. Official BGRL uses that solver plus a C grid search
and L2-normalized embeddings. We keep the official BGRL protocol because the
paper says the SSL setup follows Thakoor et al. (2022).
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize

from .compat import one_hot_encoder


def fit_logistic_regression(
    X,
    y,
    data_random_seed=1,
    repeat=1,
    n_jobs=5,
):
    encoder = one_hot_encoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).astype(bool)
    X = normalize(X, norm="l2")

    rng = np.random.RandomState(data_random_seed)
    accuracies = []
    best_val = None

    for _ in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.8, random_state=rng
        )
        logreg = LogisticRegression(solver="liblinear")
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(
            estimator=OneVsRestClassifier(logreg),
            param_grid=dict(estimator__C=c),
            n_jobs=n_jobs,
            cv=cv,
            verbose=0,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
        accuracies.append(metrics.accuracy_score(y_test, y_pred))
        best_val = float(clf.best_score_)

    return accuracies, best_val
