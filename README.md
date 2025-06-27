# Task 4 - Logistic Regression

**Objective:**  
Build a binary classification model using Logistic Regression and evaluate its performance on a medical dataset.

## ✅ Steps Performed:

### 1. Data Preprocessing
- Loaded `data.csv` and dropped missing values.
- Split the data into features (`X`) and label (`y`).

### 2. Train-Test Split
- Used 80/20 split for training and testing.

### 3. Feature Scaling
- Standardized numerical features using `StandardScaler`.

### 4. Model Training
- Trained a `LogisticRegression` model from `sklearn.linear_model`.

### 5. Evaluation Metrics
- **Confusion Matrix** – Shows TP, FP, TN, FN.
- **Precision** – TP / (TP + FP)
- **Recall** – TP / (TP + FN)
- **ROC AUC Score** – Area under ROC curve.

### 6. Visualization
- Plotted **ROC Curve** and saved as `roc_curve.png`.

## 🛠 Tools Used:
- Python
- pandas
- scikit-learn
- matplotlib


