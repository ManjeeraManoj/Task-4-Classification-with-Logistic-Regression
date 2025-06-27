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

output :

Available columns: ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

Confusion Matrix:
 [[70  1]
 [ 2 41]]
Precision: 0.98
Recall: 0.95
ROC AUC Score: 1.00

