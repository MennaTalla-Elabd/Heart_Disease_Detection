import joblib
import numpy as np

# Load Model

model = joblib.load("C:/Users/M/Downloads/heart_project/decision_tree_model.pkl")


# Prediction Function

def predict(data):
    data = np.array(data).reshape(1, -1)
    return model.predict(data)[0]

# Example Run

if __name__ == "__main__":
    sample = [0, 168, 2, 52, 1, 3, 2, 1.0, 0]  
    result = predict(sample)

    if result == 1:
        print(" High Risk")
    else:
        print(" Low Risk")