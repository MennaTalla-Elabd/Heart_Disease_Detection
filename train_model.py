import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# 1. Load Data

df = pd.read_csv("C:/Users/M/Downloads/heart_project/cleaned_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

# 2. Train/Test Split (80/20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Hyperparameter Tuning 

param_grid = {
    "max_depth": [1, 2, 3, 4],
    "min_samples_split": [5, 10, 15],
    "min_samples_leaf": [1, 2, 5, 10]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_


# 4. Evaluation

y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 5. Save Model

joblib.dump(best_model, "decision_tree_model.pkl")

print("\nModel saved successfully!")

print("Train accuracy:", best_model.score(X_train, y_train))
print("Test accuracy:", best_model.score(X_test, y_test))