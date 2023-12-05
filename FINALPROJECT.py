#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load COVID-19 data
file_path = r'C:\Users\celes\Desktop\LA_County_COVID_Cases_20231018.csv'
df = pd.read_csv(file_path)

# Sidebar for data exploration
st.sidebar.title("Data Exploration")

# Show basic info about the DataFrame
st.sidebar.subheader("DataFrame Info")
st.sidebar.write(df.info())

# Show the first few rows of the DataFrame
st.sidebar.subheader("First Few Rows")
st.sidebar.write(df.head())

# Show the columns of the DataFrame
st.sidebar.subheader("Columns")
st.sidebar.write(df.columns)

# Handle missing values
df = df.fillna(0)

# Remove duplicates
df = df.drop_duplicates()

# Display DataFrame after preprocessing
st.sidebar.subheader("Processed DataFrame")
st.sidebar.write(df.head())

# Main content
st.title("COVID-19 Data Analysis")

# Show correlation matrix heatmap
st.subheader("Correlation Matrix Heatmap")
correlation_matrix = df.corr()
st.pyplot(plt.figure(figsize=(10, 8)))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
st.pyplot()

# Histogram for new cases and new deaths
st.subheader("Distribution of New Cases and New Deaths")
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='new_cases', kde=True, bins=30, color='blue', label='New Cases')
sns.histplot(data=df, x='new_deaths', kde=True, bins=30, color='red', label='New Deaths')
plt.title('Distribution of New Cases and New Deaths in LA')
plt.xlabel('Counts')
plt.ylabel('Frequency')
plt.legend()
st.pyplot()

# Scatter plot for new state cases vs. new state deaths
st.subheader("Scatter Plot of New State Cases vs. New State Deaths")
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='new_state_cases', y='new_state_deaths', color='salmon')
plt.title('Scatter Plot of New State Cases vs. New State Deaths')
plt.xlabel('New State Cases')
plt.ylabel('New State Deaths')
st.pyplot()

# Pairplot for selected features
st.subheader("Pairplot of Selected Features")
sns.pairplot(df[['new_cases', 'new_state_cases', 'new_deaths', 'new_state_deaths']])
st.pyplot()

# Linear regression model
st.subheader("Linear Regression Model")
target_variable = 'new_deaths'
features = ['new_cases', 'state_deaths', 'state_cases']
X = df[features]
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f'Mean Absolute Error: {mae}')
st.write(f'R-squared: {r2}')

# Random Forest Classifier
st.subheader("Random Forest Classifier")
target_variable_rf = 'new_cases'
features_rf = ['new_deaths', 'new_state_deaths', 'new_state_cases']
X_rf = df[features_rf]
y_rf = df[target_variable_rf]
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
classifier = RandomForestClassifier()
classifier.fit(X_train_rf, y_train_rf)
y_pred_rf = classifier.predict(X_test_rf)
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
classification_report_rf = classification_report(y_test_rf, y_pred_rf)
st.write(f'Model Accuracy: {accuracy_rf:.2%}')
st.write("Classification Report:")
st.write(classification_report_rf)
