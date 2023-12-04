#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
desktop_path = r'C:\Users\celes\Desktop'
file_name = 'LA_County_COVID_Cases_20231018.csv'
file_path = f'{desktop_path}\\{file_name}'
df = pd.read_csv(file_path)

# Display DataFrame and info
st.write("## Data Overview")
st.write(df.head())
st.write("### Data Information")
st.write(df.info())

# Handling missing values and duplicates
st.write("### Data Cleaning")
df = df.fillna(0)
df = df.drop_duplicates()

# Display cleaned DataFrame
st.write("### Cleaned Data")
st.write(df.head())

# Correlation matrix heatmap
st.write("## Correlation Matrix")
correlation_matrix = df.corr()
st.write(sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5))
st.pyplot()

# Histogram of new cases and new deaths
st.write("## Distribution of New Cases and New Deaths")
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(data=df, x='new_cases', kde=True, bins=30, color='blue', label='New Cases', ax=ax)
sns.histplot(data=df, x='new_deaths', kde=True, bins=30, color='red', label='New Deaths', ax=ax)
plt.title('Distribution of New Cases and New Deaths in LA')
plt.xlabel('Counts')
plt.ylabel('Frequency')
plt.legend()
st.pyplot(fig)

# Scatter plot of new state cases vs. new state deaths
st.write("## Scatter Plot of New State Cases vs. New State Deaths")
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=df, x='new_state_cases', y='new_state_deaths', color='salmon', ax=ax)
plt.title('Scatter Plot of New State Cases vs. New State Deaths')
plt.xlabel('New State Cases')
plt.ylabel('New State Deaths')
st.pyplot(fig)

# Pairplot
st.write("## Pair Plot")
sns.pairplot(df[['new_cases', 'new_state_cases', 'new_deaths', 'new_state_deaths']])
st.pyplot()

# Linear regression model and evaluation
st.write("## Linear Regression Model")
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

# Visualizations using Streamlit
st.write("## Streamlit Visualizations")
st.line_chart(df.set_index('date')[['DailyCases_LA', 'DailyDeaths_LA']])






























