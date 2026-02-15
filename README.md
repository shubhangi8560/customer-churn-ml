# customer-churn-ml
#Machine learning assignment focussing on real world deployment

A. Problem statement
---
This dataset contains details of a bank's customers and their status—whether they have stayed with the bank or exited. Customer churn (attrition) is a critical metric for banks because it is significantly more expensive to acquire a new customer than to retain an existing one. This dataset is designed to help build predictive models that identify "at-risk" customers.
The goal is to analyze customer demographics and financial behavior to predict Churn (Target Variable: Exited). By identifying these patterns, the bank can proactively offer incentives, better services, or personalized financial products to improve retention.
We study various models on the chosen dataset


B. Dataset description
---
The dataset used is the “Bank Customer Churn Prediction” dataset obtained from Kaggle.
The dataset contains approximately 10,000 customer records with the following types of features:
- CreditScore – Customer credit score
- Geography – Customer location
- Gender – Male/Female
- Age – Customer age
- Tenure – Number of years with bank
- Balance – Account balance
- NumOfProducts – Number of bank products used
- HasCrCard – Whether customer has a credit card
- IsActiveMember – Active membership status
- EstimatedSalary – Estimated yearly salary

Non-informative columns such as RowNumber, CustomerId, and Surname were removed before training.
The dataset satisfies:
- Minimum 500 instances 
- Minimum 12 features 


C. Models used:
----

|ML model name |Accuracy     |AUC  |Precision  |Recall  |F1 Score     |MCC |
|--------------|-------------|-----|-----------|--------|-------------|----|
|Logistic Regression    |0.8050  |0.7710     |0.5859  |0.1425    |0.2292  |0.2167  |
|Decision Tree          |0.7755  |0.6643     |0.4512  |0.4767    |0.4636  |0.3219|
|KNN                    |0.8350  |0.7724     |0.6624  |0.3857    |0.4876  |0.4180|
|Naive Bayes            |0.8290  |0.8146|     0.7559|  0.2359|    0.3596|  0.3573|
|Random Forest          |0.8645  |0.8469|     0.7857|  0.4595|    0.5798|  0.5315|
|XGBoost                |0.8470  |0.8330|     0.6784|  0.4717|    0.5565|  0.4789|

Observations on each model:
|ML Model name		    |Observations|
|---------------------|------------|
|Logistic regression	|Performed reasonably well but struggled to capture complex non-linear relationships in the dataset|
|Decision Tree		    |Captured non-linear patterns but showed signs of overfitting compared to ensemble models|
|KNN					        |Moderate performance; sensitive to scaling and choice of neighbors|
|Naive Bayes			    |Lower performance due to assumption of feature independence, which may not hold for banking data|
|Random Forest		    |Significantly improved performance by reducing overfitting and capturing feature interactions|
|XGBoost				      |Achieved the best overall performance with highest AUC and MCC due to gradient boosting and better optimisation|


