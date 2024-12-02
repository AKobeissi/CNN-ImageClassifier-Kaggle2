import numpy as np
import pickle
import pandas as pd
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels"""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def _split_data(self, X, y, feature_idx, threshold):
        """Split data based on a feature and threshold"""
        left_mask = X[:, feature_idx] <= threshold
        return (X[left_mask], y[left_mask], 
                X[~left_mask], y[~left_mask])

    def _find_best_split(self, X, y):
        """Find the best split for a node"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        current_impurity = self._gini_impurity(y)
        
        # Sample sqrt(n_features) features randomly
        n_features_to_sample = int(np.sqrt(n_features))
        feature_indices = np.random.choice(n_features, n_features_to_sample, replace=False)
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            if len(thresholds) > 10:  # Sample thresholds for efficiency
                thresholds = np.random.choice(thresholds, 10, replace=False)
                
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, threshold)
                
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue
                    
                # Calculate information gain
                p_left = len(y_left) / len(y)
                p_right = len(y_right) / len(y)
                gain = current_impurity - (p_left * self._gini_impurity(y_left) + 
                                        p_right * self._gini_impurity(y_right))
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples_per_class = np.bincount(y)
        predicted_class = np.argmax(n_samples_per_class)
        
        node = {'class': predicted_class}
        
        # Stop if max_depth reached or node is pure
        if (depth >= self.max_depth or 
            len(np.unique(y)) == 1 or 
            len(y) < self.min_samples_split):
            return node
        
        feature_idx, threshold, gain = self._find_best_split(X, y)
        
        if feature_idx is None or gain <= 0:
            return node
        
        # Split the data
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, threshold)
        
        # Create child nodes
        node['feature_idx'] = feature_idx
        node['threshold'] = threshold
        node['left'] = self._build_tree(X_left, y_left, depth + 1)
        node['right'] = self._build_tree(X_right, y_right, depth + 1)
        
        return node

    def fit(self, X, y):
        """Train the decision tree"""
        self.tree = self._build_tree(X, y)

    def _predict_single(self, x, node):
        """Predict class for a single sample"""
        if 'feature_idx' not in node:
            return node['class']
            
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        return self._predict_single(x, node['right'])

    def predict(self, X):
        """Predict classes for multiple samples"""
        return np.array([self._predict_single(x, self.tree) for x in X])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        """Train the random forest"""
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Train tree
            tree = DecisionTree(max_depth=self.max_depth, 
                              min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        """Predict classes for multiple samples"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Return most common prediction for each sample
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] 
                        for i in range(X.shape[0])])

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    """Load and preprocess the data"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Flatten the images
    X = np.array([img.flatten() for img in data['images']])
    
    # Normalize the pixel values
    X = X / 255.0
    
    if 'labels' in data:
        y = np.array(data['labels'])
        return X, y
    return X

# Training and prediction script
def train_and_predict(train_file, test_file, sample_size=10000):
    # Load training data
    X_train, y_train = load_and_preprocess_data(train_file)
    
    # Sample subset of training data
    if sample_size and sample_size < len(X_train):
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    # Initialize and train the random forest
    rf = RandomForest(n_trees=10, max_depth=10, min_samples_split=5)
    rf.fit(X_train, y_train)
    
    # Load and preprocess test data
    X_test = load_and_preprocess_data(test_file)
    
    # Make predictions
    predictions = rf.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': range(1, len(predictions) + 1),
        'Class': predictions
    })
    
    # Save predictions
    submission.to_csv('random_forest_predictions4.csv', index=False)
    return rf, submission


# Train the model and generate predictions
rf_model, predictions = train_and_predict(
    train_file=r'data\train_data.pkl',
    test_file=r'data\test_data.pkl',
    sample_size=100000  # Adjust this based on your computational resources
)
