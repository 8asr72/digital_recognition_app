from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

print("Loading MNIST dataset...")

# Load MNIST dataset (avoid pandas dependency)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data
y = mnist.target

# Normalize data (VERY IMPORTANT for faster convergence)
X = X / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optimized Logistic Regression
model = LogisticRegression(
    max_iter=300,          # 300 is enough
    solver='saga',         # faster for large datasets
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")
