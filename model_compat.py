import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier


RANDOM_STATE = 2026


class TopKGradientBoostingSelector(BaseEstimator, TransformerMixin):

    def __init__(self, k=5, random_state=RANDOM_STATE):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=float)

        estimator = GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=3,
            random_state=self.random_state,
        )

        estimator.fit(
            X_array,
            np.asarray(y, dtype=int)
        )

        self.feature_importances_ = np.asarray(
            estimator.feature_importances_,
            dtype=float
        )

        k = min(
            max(int(self.k), 1),
            X_array.shape[1]
        )

        self.selected_indices_ = np.argsort(
            -self.feature_importances_
        )[:k]

        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_indices_]

    def get_support(self, indices=False):

        if indices:
            return self.selected_indices_

        mask = np.zeros(
            len(self.feature_importances_),
            dtype=bool
        )

        mask[self.selected_indices_] = True

        return mask