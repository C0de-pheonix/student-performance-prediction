import numpy as np

class CustomLinearRegression:
    """Linear Regression from scratch using Ordinary Least Squares (OLS) with L2 regularization (Ridge)."""
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Adding a small L2 penalty to ensure matrix is always invertible
        n_features = X.shape[1]
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        lambda_val = 0.01 
        I = np.eye(X_b.shape[1])
        I[0, 0] = 0 # Don't regularize the bias term
        
        theta = np.linalg.inv(X_b.T.dot(X_b) + lambda_val * I).dot(X_b.T).dot(y)
        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        return X.dot(self.weights) + self.bias

class Node:
    """Decision Tree Node."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, probas=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probas = probas

class CustomDecisionTreeClassifier:
    """Decision Tree Classifier from scratch using Gini impurity."""
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_classes_ = None
        self.feature_importances_ = None
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root = self._grow_tree(X, y, depth=0)
        
        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes_)
        probabilities = counts / m
        return 1.0 - np.sum(probabilities ** 2)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return self._create_leaf(y)

        best_feat, best_thresh, best_gain = self._best_split(X, y, n_features)

        if best_gain <= 0:
            return self._create_leaf(y)

        # Update feature importance
        self.feature_importances_[best_feat] += best_gain * n_samples
        
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh
        
        if np.sum(left_idxs) < self.min_samples_leaf or np.sum(right_idxs) < self.min_samples_leaf:
            return self._create_leaf(y)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(feature_idx=best_feat, threshold=best_thresh, left=left, right=right)
        
    def _create_leaf(self, y):
        counts = np.bincount(y, minlength=self.n_classes_)
        leaf_value = np.argmax(counts)
        probas = counts / np.sum(counts)
        return Node(value=leaf_value, probas=probas)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        best_feat, best_thresh = None, None
        parent_gini = self._gini(y)
        
        # Optimize by only checking percentiles instead of every value
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.percentile(X_column, [25, 50, 75])
            
            for threshold in np.unique(thresholds):
                left_idxs = X_column <= threshold
                right_idxs = X_column > threshold
                
                n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
                if n_left == 0 or n_right == 0:
                    continue
                    
                gini_left = self._gini(y[left_idxs])
                gini_right = self._gini(y[right_idxs])
                
                gain = parent_gini - (n_left / len(y) * gini_left + n_right / len(y) * gini_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = threshold
                    
        return best_feat, best_thresh, best_gain

    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value, node.probas
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        predictions = [self._predict_single(x, self.root)[0] for x in X]
        return np.array(predictions)
        
    def predict_proba(self, X):
        probas = [self._predict_single(x, self.root)[1] for x in X]
        return np.array(probas)


class CustomRandomForestClassifier:
    """Random Forest Classifier from scratch using Bootstrap Aggregation."""
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=None, n_jobs=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        feature_importances = np.zeros(n_features)
            
        for _ in range(self.n_estimators):
            tree = CustomDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            
            # Bootstrap sample
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            if tree.feature_importances_ is not None:
                feature_importances += tree.feature_importances_
            
        self.feature_importances_ = feature_importances / self.n_estimators

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        majority_votes = []
        for i in range(X.shape[0]):
            counts = np.bincount(tree_preds[:, i], minlength=self.n_classes_)
            majority_votes.append(np.argmax(counts))
        return np.array(majority_votes)
        
    def predict_proba(self, X):
        tree_probas = np.array([tree.predict_proba(X) for tree in self.trees])
        # Average probabilities
        return np.mean(tree_probas, axis=0)
