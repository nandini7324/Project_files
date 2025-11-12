# Project 3 - Sales Data Analysis
# Analysis of monthly sales data and visualize trends.

import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'Month': ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    'Sales': [12000, 15000, 18000, 17000, 20000, 21000, 22000, 19000, 25000, 23000, 24000, 26000]
}
df = pd.DataFrame(data)

# Summary
print("Sales Summary:\n", df.describe())

# Highest and lowest month
max_month = df.loc[df['Sales'].idxmax(), 'Month']
min_month = df.loc[df['Sales'].idxmin(), 'Month']
print(f"\nHighest sales in {max_month}, Lowest sales in {min_month}")

# Plot
plt.plot(df['Month'], df['Sales'], marker='o', color='green')
plt.title("Monthly Sales Analysis")
plt.xlabel("Month")
plt.ylabel("Sales Amount")
plt.grid(True)
plt.show()
