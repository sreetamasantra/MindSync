import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load data 

df = pd.read_csv("data/raw/session_data.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

# Features and labels 

FEATURES = ["ear", "pitch_adj", "yaw_adj", "blink_rate", "closed_frames"]
X = df[FEATURES].values
y = df["label"].values

# Encode labels to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nClasses: {le.classes_}")

# Train/test split 

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2,
    random_state=42, stratify=y_encoded
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# Train Random Forest 

print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)
print("Training complete!")

# Evaluate 

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average="weighted")

print(f"\n{'='*40}")
print(f"Accuracy:    {accuracy*100:.2f}%")
print(f"F1 Score:    {f1*100:.2f}%")
print(f"{'='*40}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_))

# Cross validation 

print("Running 5-fold cross validation...")
cv_scores = cross_val_score(model, X, y_encoded, cv=5)
print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% "
      f"(+/- {cv_scores.std()*100:.2f}%)")

# Feature importance 

print("\nFeature Importances:")
for name, imp in sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 50)
    print(f"  {name:15} {imp:.4f}  {bar}")

# Confusion Matrix plot 

os.makedirs("data/processed", exist_ok=True)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")
plt.title("Confusion Matrix — MindSync Cognitive State Classifier")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("data/processed/confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved to data/processed/confusion_matrix.png")

# Save model 

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/cognitive_classifier.pkl")
joblib.dump(le,    "models/label_encoder.pkl")
print("Model saved to models/cognitive_classifier.pkl")
print("Label encoder saved to models/label_encoder.pkl")