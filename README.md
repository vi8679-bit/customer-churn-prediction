# Customer Churn Prediction using Machine Learning

## Overview

Customer churn prediction is a common problem in business analytics where the goal is to identify customers who are likely to stop using a service. Predicting churn helps companies take proactive steps to retain customers and reduce revenue loss.

This project builds a machine learning pipeline to analyze customer data and predict whether a customer will churn based on demographic, account, and service-related features.

The project demonstrates a full data science workflow including data cleaning, exploratory data analysis, feature preprocessing, model training, and model evaluation.


## Problem Statement

Customer retention is a major challenge for subscription-based businesses. When customers leave a service, companies lose recurring revenue and acquisition costs increase.

The objective of this project is to develop a predictive model that classifies whether a customer is likely to churn using historical customer data.


## Dataset

This project uses a customer churn dataset containing customer demographics, service subscriptions, and billing information.

### Target Variable

Churn

* Yes - customer leaves the service
* No - customer continues using the service

### Example Features

* gender
* SeniorCitizen
* tenure
* InternetService
* Contract
* MonthlyCharges
* TotalCharges

These variables capture both demographic and behavioral aspects of customer activity.


## Exploratory Data Analysis

Exploratory data analysis was performed to understand patterns related to customer churn.

Key analyses included:

* distribution of customer tenure
* churn distribution across contract types
* relationship between monthly charges and churn
* service subscription analysis
* correlation between numerical variables

These analyses help identify the factors most strongly associated with customer attrition.

## Data Preprocessing

Before model training, several preprocessing steps were applied:

* handling missing values
* encoding categorical variables
* scaling numerical variables
* train-test split for model evaluation

These steps ensure that the dataset is suitable for machine learning algorithms.


## Machine Learning Models

Multiple supervised learning models were evaluated to predict customer churn.

Models implemented include:

* Logistic Regression
* Random Forest
* Decision Tree

These models are widely used for binary classification tasks and allow comparison between linear and tree-based approaches.


## Model Evaluation

The models were evaluated using common classification metrics:

* Accuracy
* Precision
* Recall
* F1 Score

Confusion matrices were also used to visualize classification performance and understand prediction errors.


## Key Insights

Exploratory analysis suggests several important factors influencing customer churn:

* customers with shorter tenure are more likely to churn
* higher monthly charges are associated with higher churn rates
* contract type strongly affects retention
* customers with longer service relationships tend to remain loyal

These insights can help businesses develop targeted retention strategies.



## Technologies Used

Python
pandas
NumPy
scikit-learn
Matplotlib
Seaborn



## Future Improvements

Possible improvements for this project include:

* implementing gradient boosting models such as XGBoost or LightGBM
* hyperparameter optimization
* feature importance analysis
* deployment of the model as a prediction dashboard

