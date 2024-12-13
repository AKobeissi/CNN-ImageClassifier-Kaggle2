import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
def load_data():
    with open('/kaggle/input/ift3395-ift6390-identification-maladies-retine/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    return train_data

# Preprocess the data
def preprocess_data(train_data, subset_size=50000):
    # Convert data to numpy arrays and normalize
    X = np.array(train_data['images'], dtype=np.float32) / 255.0
    y = np.array(train_data['labels'])

    # Subset the training data
    subset_indices = np.random.choice(len(X), subset_size, replace=False)
    X = X[subset_indices]
    y = y[subset_indices]

    # Flatten images (convert to 1D array per image)
    X = X.reshape(X.shape[0], -1)

    return X, y

# Train and evaluate models
def train_and_evaluate(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    # Train the model
    print(f"Training {model_name} on {X_train.shape[0]} samples...")
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy for {model_name}: {val_accuracy:.4f}")

    # Classification report on validation set
    val_class_report = classification_report(y_val, val_predictions, output_dict=True)
    print(f"Classification Report for {model_name} on Validation Set:\n")
    print(classification_report(y_val, val_predictions))

    # Evaluate on test set
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test Accuracy for {model_name}: {test_accuracy:.4f}")

    # Classification report on test set
    test_class_report = classification_report(y_test, test_predictions, output_dict=True)
    print(f"Classification Report for {model_name} on Test Set:\n")
    print(classification_report(y_test, test_predictions))

    return val_accuracy, test_accuracy, val_class_report, test_class_report

if __name__ == "__main__":
    # Load and preprocess data
    train_data = load_data()
    X, y = preprocess_data(train_data, subset_size=50000)

    # Split into train+val and test (90%-10%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # Split train+val into training and validation sets (70%-20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2222, random_state=42, stratify=y_train_val
    )

    print(f"Training Set Size: {X_train.shape[0]} samples")
    print(f"Validation Set Size: {X_val.shape[0]} samples")
    print(f"Test Set Size: {X_test.shape[0]} samples")

    # Initialize models
    models = {
        'SVM': SVC(kernel='rbf', C=1, gamma='scale', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = []
    classification_reports = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        val_accuracy, test_accuracy, val_class_report, test_class_report = train_and_evaluate(
            model, model_name, X_train, y_train, X_val, y_val, X_test, y_test
        )

        # Store results
        results.append({'Model': model_name, 'Validation Accuracy': val_accuracy, 'Test Accuracy': test_accuracy})
        classification_reports[model_name] = {
            'Validation': val_class_report,
            'Test': test_class_report
        }

    # Display results
    results_df = pd.DataFrame(results)
    print("\nResults Summary:")
    print(results_df)

    # Save results for the results section
    results_df.to_csv('model_results_summary.csv', index=False)
    print("Results summary saved as 'model_results_summary.csv'.")

    # Save classification reports
    with open('classification_reports.pkl', 'wb') as f:
        pickle.dump(classification_reports, f)
    print("Classification reports saved as 'classification_reports.pkl'.")
