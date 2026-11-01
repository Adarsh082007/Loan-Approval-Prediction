# Loan Approval Prediction using Machine Learning

## Project Overview

This project focuses on predicting whether a loan application will be approved or rejected using Machine Learning classification algorithms.

The dataset contains 1,000 records with 20 columns, including applicant income, co-applicant income, employment status, age, credit score, loan amount, loan term, education level, and other financial details.

The main objective is to analyze applicant information and build machine learning models to predict loan approval.

## Technologies & Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Project Workflow

### 1. Data Loading & Inspection

* Loaded dataset using Pandas.
* Inspected dataset using `head()` and `info()`.
* Checked data types and missing values.

### 2. Data Cleaning

* Identified missing values.
* Handled categorical missing values using SimpleImputer with the most frequent strategy.
* Handled numerical missing values using SimpleImputer with the mean strategy.
* Removed unnecessary columns such as Applicant_ID.

### 3. Exploratory Data Analysis (EDA)

Performed data analysis and visualization using:

* Pie Chart – Loan Approval Distribution.
* Bar Chart – Education Level Distribution.
* Bar Chart – Gender Distribution.
* Histogram – Applicant Income Distribution.
* Histogram – Credit Score Distribution based on Loan Approval.

### 4. Data Preprocessing

* Applied Label Encoding to categorical features.
* Applied One Hot Encoding to convert categorical variables into numerical format.
* Combined encoded features with numerical features.

### 5. Train-Test Split

* Split dataset into 80% training data and 20% testing data.
* Used `random_state=42`.

### 6. Feature Scaling

* Applied StandardScaler to standardize numerical features.
* Used `fit_transform()` on training data.
* Used `transform()` on testing data.

### 7. Machine Learning Algorithms

Implemented and trained the following classification algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes

### 8. Model Evaluation

Evaluated model performance using:

* Accuracy Score
* Precision Score
* Recall Score
* Confusion Matrix

## Model Performance

| Algorithm            | Accuracy |
| -------------------- | -------: |
| Logistic Regression  |    86.5% |
| K-Nearest Neighbors  |    74.5% |
| Gaussian Naive Bayes |    86.5% |

### Detailed Evaluation Results

| Algorithm            | Precision | Recall |
| -------------------- | --------: | -----: |
| Logistic Regression  |    78.33% | 77.05% |
| KNN                  |    60.87% | 45.90% |
| Gaussian Naive Bayes |    80.35% | 73.77% |

## Key Findings

* Logistic Regression and Gaussian Naive Bayes achieved the highest accuracy of 86.5%.
* KNN achieved an accuracy of 74.5%.
* Gaussian Naive Bayes achieved the highest precision among the three models.
* Credit Score and applicant financial information were explored during EDA.