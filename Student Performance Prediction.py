# Student Performance Prediction Project

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # 1. Created a synthetic dataset
    np.random.seed(42)
    data = {
        'Study_Hours': np.random.randint(1, 10, 100),
        'Attendance': np.random.randint(60, 100, 100),  
        'Internal_Marks': np.random.randint(20, 30, 100),
        'Final_Exam_Score': np.random.randint(35, 100, 100)
    }

    df = pd.DataFrame(data)
    print("Sample Data:")
    print(df.head())

    # 2. Data exploration
    print("\nDataset Information:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

    # Correlation heatmap
    plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    # 3. Data cleaning
    print("\nChecking for missing values:")
    print(df.isnull().sum())

    df = df.dropna()
    df = df.drop_duplicates()

    # 4. Splitting data into train and test sets
    X = df[['Study_Hours', 'Attendance', 'Internal_Marks']]
    y = df['Final_Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nModel coefficients:", dict(zip(X.columns, model.coef_)))
    print("Model intercept:", model.intercept_)

    # Predictions
    y_pred = model.predict(X_test)

    # 6. Model evaluation
    print("\nModel Evaluation Metrics:")
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    # 7. Visualization of results
    plt.scatter(y_test, y_pred, color='blue')
    plt.xlabel("Actual Scores")
    plt.ylabel("Predicted Scores")
    plt.title("Actual vs Predicted Student Performance")
    plt.tight_layout()
    plt.show()

    sns.histplot((y_test - y_pred), kde=True)
    plt.title('Error Distribution')
    plt.tight_layout()
    plt.show()

    # 8. Predicting for a new student
    new_data = pd.DataFrame({
        'Study_Hours': [8],
        'Attendance': [92],  # renamed column
        'Internal_Marks': [27]
    })

    # Ensure column order and dtypes match training data
    new_data = new_data[X.columns]
    # cast types to match training features
    new_data = new_data.astype(X.dtypes.to_dict())

    predicted_score = model.predict(new_data)
    print("\nPredicted Final Exam Score for new student:", float(predicted_score[0]))

if __name__ == "__main__":
    main()
