# ❤️ Heart Disease Prediction using Logistic Regression

## Project Overview

> A model that confidently identifies healthy patients can be just as valuable as one that detects disease.

This project explores how Logistic Regression can be used to predict the presence of heart disease using patient health indicators from the UCI Heart Disease dataset.

Rather than relying on a single biomarker, the model analyzes demographic, physiological, and clinical measurements to estimate cardiovascular disease risk and uncover clinically meaningful performance patterns.

The project implements a complete machine learning workflow including:

- Data preprocessing
- Missing value handling
- Exploratory data analysis
- Feature correlation analysis
- Logistic Regression model training
- Classification evaluation
- Deployment-ready model serialization using Joblib

The final model demonstrates how interpretable machine learning can support early-risk screening and prevention-focused healthcare strategies.

---

# 🔍 Key Clinical Insights

## ✅ 1. High Clinical Reliability for Healthy Patients (Rule-Out Tool)

### The Insight
The model performs exceptionally well at identifying patients who do **not** have heart disease.

It achieved:

- **90% Recall** for healthy patients (Class 0)
- **87% Precision** for healthy patients

### Why This Matters
This means the model can potentially function as a reliable **rule-out screening tool** during early clinical triage.

If the model predicts a patient as healthy, clinicians can have relatively high confidence in that prediction, helping:

- Reduce unnecessary diagnostic testing
- Lower hospital resource utilization
- Improve triage efficiency
- Prioritize higher-risk patients more quickly

---

## ⚠️ 2. High Risk of Severe Misclassification (Class 3 vs. Class 1)

### The Insight
The classification report revealed that the model struggles heavily to distinguish **mild heart disease (Class 1)** from more advanced disease stages.

Performance imbalance included:

- Extremely low recall (**8%**) for Class 1
- Much stronger recall (**71%**) for severe disease cases (Class 3)

### Why This Matters
This represents a major clinical limitation.

The model is prone to confusing:
- early-stage disease
- advanced disease
- and occasionally healthy patients

In a real healthcare environment, this could lead to:
- delayed intervention
- incorrect treatment prioritization
- missed preventative care opportunities

This demonstrates why evaluating only overall accuracy can be misleading in healthcare machine learning systems.

---

## 📈 3. Actionable Strategic Improvement — Binary Classification Framework

### The Insight
The dataset contains only **303 total patient records** distributed across **5 separate disease classes**.

This creates severe class fragmentation and weak statistical density per class, which directly contributes to the model’s modest overall accuracy (~54%).

### Why This Matters
A stronger production strategy would be to redesign the problem into:

# Presence vs. Absence of Heart Disease

instead of predicting the exact disease stage.

By grouping:
- Classes 1–4 → “Disease Present”
- Class 0 → “No Disease”

the model would gain:
- stronger class balance
- larger effective sample sizes
- improved statistical learning stability
- significantly better predictive reliability

This shift would make the model substantially safer and more realistic for real-world clinical decision support systems.

---

# 🧠 What is Logistic Regression?

Logistic Regression is a supervised machine learning algorithm designed for classification problems.

Instead of predicting continuous numeric values, the algorithm predicts the probability that an observation belongs to a particular class.

In this project, the model predicts whether a patient is likely to have heart disease.

## Sigmoid Function

```text
P(Y=1) = 1 / (1 + e^(-z))
```

Where:

```text
z = b0 + b1x1 + b2x2 + ... + bnxn
```

The output ranges between:
- 0 → low probability
- 1 → high probability

---

# 📊 Dataset

## Source
UCI Machine Learning Repository

## Features Include

- Age
- Sex
- Chest pain type (`cp`)
- Resting blood pressure (`trestbps`)
- Cholesterol (`chol`)
- Fasting blood sugar (`fbs`)
- ECG results (`restecg`)
- Maximum heart rate (`thalach`)
- Exercise-induced angina (`exang`)
- ST depression (`oldpeak`)
- Slope
- Number of vessels (`ca`)
- Thalassemia (`thal`)

## Target Variable

`num` → Presence of heart disease

---

# 🛠️ Machine Learning Pipeline

## 1. Data Cleaning & Preprocessing

The dataset was cleaned and prepared prior to model training, including:

- Missing value handling
- Data type inspection
- Feature scaling using `StandardScaler`
- Train-test splitting

---

## 2. Exploratory Data Analysis

Exploratory analysis was performed to identify trends and feature relationships within the dataset.

### Correlation Heatmap

A Pearson correlation heatmap was used to inspect relationships between variables and identify potentially important predictors.

### Age Distribution

Age distributions revealed that most patients fall between 40 and 70 years old, aligning with known cardiovascular risk patterns.

---

# 🤖 Logistic Regression Model Training

The model was trained using Scikit-learn’s `LogisticRegression` implementation.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

Feature scaling was applied before training to improve optimization stability and coefficient convergence.

---

# 📈 Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Confusion Matrix

The confusion matrix compares predicted classifications against actual labels and provides insight into:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

This evaluation helps measure how effectively the model distinguishes between healthy and high-risk patients.

---

# 💾 Deployment Preparation

To support future deployment and real-time inference workflows, trained model artifacts were serialized using Joblib.

```python
import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/logistic_regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
```

This allows the trained model and preprocessing pipeline to be reused later without retraining the notebook every time.

---

# 📂 Project Structure

```bash
heart-disease-logistic-regression/
│
├── LogisticRegression.ipynb
├── README.md
├── requirements.txt
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── scaler.pkl
│
└── images/
    ├── age_distribution.png
    ├── correlation_heatmap.png
    └── confusion_matrix.png
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/heart-disease-logistic-regression
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook LogisticRegression.ipynb
```

Run all notebook cells to:
- preprocess the dataset
- train the model
- evaluate performance
- generate visualizations

---

# 🚀 Future Improvements

Potential next-stage enhancements include:

- Hyperparameter tuning
- Feature engineering
- ROC-AUC analysis
- Streamlit deployment
- Real-time prediction interface
- Additional classification models

---

# 📌 Key Takeaways

- Logistic Regression provides interpretable classification for healthcare prediction tasks.
- Feature scaling improves optimization stability.
- Heal
