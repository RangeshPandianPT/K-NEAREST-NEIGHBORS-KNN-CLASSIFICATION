# KNN Classifier on Iris Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
df = pd.read_csv("Iris.csv")
df.drop(columns=["Id"], inplace=True)
X = df.drop("Species", axis=1)
y = df["Species"]
X_scaled = StandardScaler().fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train and evaluate KNN
k_values = range(1, 16)
accuracies = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Final model
best_k = k_values[np.argmax(accuracies)]
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, final_pred))
print("Best K:", best_k)
print("Accuracy:", accuracy_score(y_test, final_pred))
