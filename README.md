# Loan Prediction using Multiple Classifiers

This project focuses on predicting whether a loan applicant will default on their loan or not based on various applicant features. We implemented multiple machine learning models to predict loan approval and used different techniques to evaluate their performance.

The models used in this project are:

- XGBoost

- Random Forest Classifier

- Decision Tree Classifier

- K-Nearest Neighbors Classifier (KNN)

- Logistic Regression

- Gaussian Naive Bayes

- CATBoost Algorithm

The goal of this project is to evaluate the performance of these classifiers on the loan prediction task and identify the most effective model.

## Project Overview
Loan approval prediction is a classification problem where the goal is to predict whether an applicant will be approved for a loan based on various features, such as:

- Applicant’s age

- Employment status

- Credit history

- Loan amount requested

- Income level

- Property area

- Education level

- Marital status

The models built in this project were evaluated on the basis of their ability to accurately predict whether an applicant will default on the loan (i.e., predict whether the applicant is classified as approved or denied).

## Technologies Used
- Python 3.x

- Pandas, NumPy (for data manipulation)

- Scikit-learn (for model training and evaluation)

- XGBoost (for XGBoost model)

- CatBoost (for CatBoost model)

- Matplotlib, Seaborn (for data visualization)

- Jupyter Notebook (for analysis and visualization)

## Model Evaluation
- Accuracy: The proportion of correctly predicted instances out of all instances.

- Precision: The proportion of positive instances correctly identified by the model.

- Recall: The proportion of actual positive instances correctly identified.

- F1-Score: The harmonic mean of precision and recall, providing a balance between the two.

## Performance Metrics
- Each classifier was evaluated based on accuracy, precision, recall, and F1-score.

- XGBoost and CatBoost performed well due to their ability to handle complex datasets.

- Random Forest and Logistic Regression also provided strong results, but Decision Tree and KNN performed slightly worse on this dataset.
