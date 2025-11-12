# Project 2- Data Cleaning and Analysis 
# city whether report analysis

import pandas as pd
import matplotlib.pyplot as plt

# Messy dataset
data = {
    'City': ['Mumbai', 'Delhi', 'Pune', 'Chennai', 'Bangalore', 'Delhi'],
    'Temperature': [32, 35, None, 31, 28, None],
    'Humidity': [70, None, 60, 75, 80, 65]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# Filled missing values with column mean
df['Temperature'].fillna(df['Temperature'].mean(), inplace=True)
df['Humidity'].fillna(df['Humidity'].mean(), inplace=True)

# Remove duplicates
df = df.drop_duplicates()

print("\nCleaned Data:\n", df)

# Plot
plt.plot(df['City'], df['Temperature'], marker='o', label='Temperature')
plt.plot(df['City'], df['Humidity'], marker='s', label='Humidity')
plt.title("City Weather Report")
plt.xlabel("City")
plt.ylabel("Value")
plt.legend()
plt.show()
