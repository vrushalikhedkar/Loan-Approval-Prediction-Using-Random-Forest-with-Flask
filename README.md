# Loan Approval Prediction Using Random Forest

A Machine Learning project that predicts whether a loan application will be approved or rejected using the Random Forest Classification algorithm.


### Project Overview

This project uses customer and loan application information such as income, loan amount, credit history, education, and property area to predict the loan approval status.

The project includes data preprocessing, categorical data encoding, train-test split, Random Forest model training, model evaluation, confusion matrix visualization, new application prediction, and model saving.


### Technologies UsedPython

- Pandas
- Matplotlib
- Scikit-learn
- Pickle
- Flask


### Machine Learning Algorithm

***Random Forest Classifier***

Random Forest is an Ensemble Learning algorithm that combines multiple Decision Trees to make a final classification prediction.


*In this project:*

- 100 Decision Trees are used.
- Categorical features are converted into numerical form using One-Hot Encoding.
- The dataset is divided into training and testing data.
- The model predicts whether a loan is Approved or Rejected.
- The trained model is saved using Pickle.


### Data Preprocessing

- Check dataset shape
- Check missing values
- Remove missing rows
- Check and remove duplicate rows
- Convert categorical columns using ```pd.get_dummies()```
- Convert Loan Status: ```Y``` = Approved and ```N``` = Rejected

### Model Evaluation

The model is evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report

### Confusion Matrix

The confusion matrix shows the correct and incorrect predictions made by the model.

<img src="confusion_Matrix_grph.png" width="500">


### New Loan Application Prediction

After training, the model is used to predict the loan status of a new application.

The output can be:

- Approved <br />
- Rejected


### Model Saving

The trained Random Forest model is saved as:

```loan_approval_random_forest.pkl```

This allows the trained model to be reused later without training it again.


### Project Workflow

1.  Load the dataset

2.  Check and clean the data

3.  Encode categorical features

4.  Separate features and target

5.  Split data into training and testing sets

6.  Train Random Forest Classifier

7.  Predict test data

8.  Evaluate the model

9.  Visualize the confusion matrix

10.  Predict a new loan application

11.  Save the trained model


### Conclusion

This project demonstrates how Random Forest Classification can be used for loan approval prediction and how a trained Machine Learning model can be saved for future use or deployment.


