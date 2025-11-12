# Project 1 - Average Student Score Analysis

import pandas as pd
import matplotlib.pyplot as plt

# Created dataset
data = {
    'Name': ['Amit', 'Nina', 'Ravi', 'Kiran', 'Priya'],
    'Math': [78, 85, 62, 90, 70],
    'Science': [88, 79, 74, 95, 68],
    'English': [82, 80, 60, 85, 72]
}

df = pd.DataFrame(data)

# Calculate total and average
df['Total'] = df[['Math', 'Science', 'English']].sum(axis=1)
df['Average'] = df[['Math', 'Science', 'English']].mean(axis=1)

print("Student Performance Summary:\n", df)

# Plot
plt.bar(df['Name'], df['Average'], color='skyblue')
plt.title("Average Marks of Students")
plt.xlabel("Students")
plt.ylabel("Average Marks")
plt.show()
