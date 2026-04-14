import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Expert System

def expert_system(row):
    score = 0

    # Chest pain
    if row["cp"] == 3:
        score += 1

    # Heart rate
    if row["thalach"] < 100:
        score += 1

    # ECG indicator
    if row["oldpeak"] > 2:
        score += 1

    # Exercise angina
    if row["exang"] == 1:
        score += 1

    # Number of vessels
    if row["ca"] > 1:
        score += 1

    # Age risk
    if row["age"] > 55:
        score += 1

    # Slope risk
    if row["slope"] == 2:
        score += 1

    # Thal risk
    if row["thal"] == 3:
        score += 1

    # Final decision
    return 1 if score >= 2 else 0


df = pd.read_csv("C:/Users/M/Downloads/heart_project/data/cleaned_data.csv")

features = [
    "cp", "thalach", "slope", "age",
    "sex", "thal", "ca", "oldpeak", "exang"
]

X = df[features]
y = df["target"]


# Train / Validation Split

_, X_val, _, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load ML Model
model = joblib.load("C:/Users/M/Downloads/heart_project/decision_tree_model.pkl")

# Predictions

ml_preds = model.predict(X_val)
expert_preds = X_val.apply(expert_system, axis=1)


# Evaluation Function

def evaluate(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))



# Results

evaluate("Decision Tree", y_val, ml_preds)
evaluate("Expert System", y_val, expert_preds)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Confusion Matrix Function

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)

    plt.show()



# Decision Tree Confusion Matrix

plot_conf_matrix(y_val, ml_preds, "Decision Tree Confusion Matrix")

# Expert System Confusion Matrix

plot_conf_matrix(y_val, expert_preds, "Expert System Confusion Matrix")