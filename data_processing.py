import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/Users/M/Downloads/heart_project/data/heart.csv")

data.isnull().sum()

data.dropna(inplace=True)
data.isnull().sum()

scaler = MinMaxScaler()
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
#choose numerical features to scale
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# One-Hot Encoding
data = pd.get_dummies(data, drop_first=True)
print(data.head())

# Correlation with target
correlation = data.corr()
target_corr = correlation["target"].sort_values(ascending=False)
print(target_corr)

selected_features = target_corr[abs(target_corr) > 0.2].index
print(selected_features)

data = data[selected_features]

data.to_csv("cleaned_data.csv", index=False)