# 🌸 K-Nearest Neighbors (KNN) Classifier on Iris Dataset

This project demonstrates the implementation of the **K-Nearest Neighbors (KNN)** algorithm on the classic **Iris dataset**. It includes data preprocessing, training and evaluation with different values of **K**, performance visualization, and decision boundary plots.

---

## 📂 Project Structure

```

KNN\_Iris\_Project/
│
├── Iris.csv                    # Original dataset
├── knn\_classifier.py          # Main Python script for training & evaluation
├── knn\_accuracy\_plot.png      # Accuracy vs K plot
├── confusion\_matrix.png       # Heatmap of confusion matrix
├── decision\_boundary.png      # 2D decision boundary using two features
├── README.md                  # Project description

````

---

## 📊 Dataset

We use the **Iris dataset**, which contains:
- **150** samples
- **3 classes** of iris plant: *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*
- **4 numerical features**: Sepal Length, Sepal Width, Petal Length, Petal Width

> Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Iris)

---

## ⚙️ Tools & Libraries

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## 🚀 How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/KNN_Iris_Project.git
    cd KNN_Iris_Project
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script:
    ```bash
    python knn_classifier.py
    ```

---

## 🔍 What This Project Does

- Normalizes features using `StandardScaler`
- Splits data into train/test sets
- Trains `KNeighborsClassifier` with various K values
- Plots:
  - Accuracy vs. K
  - Confusion matrix heatmap
  - 2D decision boundary (using first two features)
- Outputs best-performing K value

---

## 📈 Visualizations


![Image](https://github.com/user-attachments/assets/2144ceda-4960-43a1-91ba-aa4bfb0e1a7f)



![Image](https://github.com/user-attachments/assets/98bc84e5-717c-46a3-b2d0-1daa1f93f7dc)



![Image](https://github.com/user-attachments/assets/c30372e6-dcc1-45aa-b24b-41e22c149257)


---

## ❓ Interview Preparation

### How does KNN work?
- It stores all training instances.
- To classify a new point, it finds the **K nearest neighbors** and assigns the majority label.

### Why normalize features?
- KNN uses distance metrics; larger-scaled features dominate unless standardized.

### How to choose K?
- Use cross-validation or accuracy plots. Too low → overfit. Too high → underfit.

### Pros and Cons of KNN
✅ Simple and effective  
❌ Computationally expensive at prediction time

### Time complexity
- **Training:** O(1)  
- **Prediction:** O(n \* d), where n = training size, d = feature dimensions

---

## 🧠 Author

RANGESHPANDIAN PT
*ML Developer | Data Science Enthusiast*

---
