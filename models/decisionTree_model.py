import numpy as np
from collections import Counter
import pandas as pd


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=150, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if self.n_features:
            self.n_features = min(X.shape[1], self.n_features)
        else:
            self.n_features = X.shape[1]

        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            counter = Counter(y)
            leaf_value = counter.most_common(1)[0][0]
            return Node(value=leaf_value)
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

        best_feature, best_threshold = self.best_split(X, y, feature_idxs)

        left_indxs, right_indxs = self.split(X[:, best_feature], best_threshold)
        left = self.grow_tree(X[left_indxs, :], y[left_indxs], depth + 1)
        right = self.grow_tree(X[right_indxs, :], y[right_indxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def best_split(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.calculate_info_gain(X_column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
        return split_idx, split_threshold

    def calculate_info_gain(self, X_column, y, threshold):
        parent_entropy = self.calculate_entropy(y)
        left_idx, right_idx = self.split(X_column, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        entropy_l, entropy_r = self.calculate_entropy(
            y[left_idx]
        ), self.calculate_entropy(y[right_idx])
        children_entropy = (n_l / n) * entropy_l + (n_r / n) * entropy_r
        info_gain = parent_entropy - children_entropy
        return info_gain

    def calculate_entropy(self, y):
        h = np.bincount(y)
        probabilities = h / len(y)
        return -np.sum([p * np.log(p) for p in probabilities if p > 0])

    def split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()

        return left_idxs, right_idxs

    def predict(self, X):
        y_ped = np.array([self.traverse_DT(x, self.root) for x in X])
        return y_ped

    def traverse_DT(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_DT(x, node.left)
        return self.traverse_DT(x, node.right)
