Diabetes Prediction Model and Streamlit Web Interface
1. Code Explanation
The diabetes prediction model is implemented using Logistic Regression, Decision Tree, and Random Forest
algorithms. The data is processed through various steps including data loading, preprocessing, splitting
into training and testing sets, and finally, model training and evaluation.
Code Sections:
1. Import Libraries: Essential libraries such as pandas, sklearn, and Streamlit are imported.
2. Load Dataset: The Pima Indians Diabetes Database is loaded for prediction.
3. Data Preprocessing: Missing values, if any, are handled, and the data is normalized.
4. Train-Test Split: The data is split into training and testing sets.
5. Model Training: Logistic Regression, Decision Tree, and Random Forest models are trained.
6. Model Evaluation: Accuracy, precision, recall, and F1 score metrics are calculated.
7. Streamlit Integration: A web interface is created to take real-time input and predict diabetes.
Code Snippet
# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
# Load the dataset
data = pd.read_csv('diabetes.csv')
# Preprocessing steps here...
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Train models
log_reg = LogisticRegression().fit(X_train, y_train)
decision_tree = DecisionTreeClassifier().fit(X_train, y_train)
random_forest = RandomForestClassifier().fit(X_train, y_train)
# Predict and evaluate
log_reg_pred = log_reg.predict(X_test)
dt_pred = decision_tree.predict(X_test)
rf_pred = random_forest.predict(X_test)
# Streamlit Interface
st.title('Diabetes Prediction App')
input_glucose = st.number_input('Enter Glucose Level')
# ... additional inputs and logic for prediction...
2. Streamlit Web Interface
The Streamlit app provides a simple web interface for users to input their health data in real time
and receive a prediction on whether they are at risk of diabetes.
Key Features:
1. Takes real-time inputs such as glucose level, BMI, age, and others.
2. Displays prediction results instantly based on the trained model.
3. User-friendly design with clear instructions for input.
Web Link: Once the app is deployed on Streamlit Cloud, users can visit the web link, enter their health parameters, and
receive instant predictions.
