"""Amazon India Warehouse Operations: StatisticalAnalysis
Company Background :Amazon India operates a large fulfillment center in Bengaluru. The HR analytics team wants to evaluate
employee working hours to optimize operations and ensure compliance with labor standards.
Population Data (All Employees)
The daily working hours (in hours) of all 25 warehouse employees on a particular day are :
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
population=np.array([8.2, 9.0, 8.7, 9.5, 10.1,
9.3, 8.9, 9.8, 10.0, 8.5,
9.1, 9.6, 8.8, 9.4, 10.2,
8.6, 9.7, 9.2, 8.4, 9.9,
10.3, 9.5, 8.9, 9.6, 8.7])

## Random sampling 
let_samplesize=8
sample=np.random.choice(population ,size=let_samplesize,replace=False)
print(sample)

## Descriptive statistics

meanP=np.mean(population)
print("population mean :",meanP)
meanS=np.mean(sample)
print("sample mean :",meanS)

medianP=np.median(population)
print("population median :",medianP)
medianS=np.median(sample)
print("sample median :",medianS)

stdP=np.std(population)
print("population standard deviation :",stdP)
stdS=np.std(sample)
print("sample standard deviation :",stdS)

maximumP=np.max(population)
print("maximum of population :",maximumP)
maximumS=np.max(sample)
print("maximum of sample :",maximumS)

minimumP=np.min(population)
print("minimum of population :",minimumP)
minimumS=np.min(sample)
print("minimum of sample :",minimumS)

## Visualization
## Histograph
plt.hist(population,bins=5)
plt.xlabel("working hours")
plt.ylabel("employee")
plt.title("population Histograph")
plt.show()

plt.hist(sample,bins=5)
plt.xlabel("working hours")
plt.ylabel("employee")
plt.title("sample Histograph")
plt.show()

# Boxplot
plt.boxplot(population)
plt.title("population Boxplot")
plt.show()

plt.boxplot(sample)
plt.title("sample Boxplot")
plt.show() 

# Q-Q plot
import scipy.stats as stats
stats.probplot(population, dist="norm", plot=plt)
plt.title("population Q-Q plot")
plt.show()

stats.probplot(sample, dist="norm", plot=plt)
plt.title("sample Q-Q plot")
plt.show()

##Claim 1 (Average Shift Hours):The average daily working hours of warehouse employees is 9 hours.
# Hypothesis Testing
# Null Hypothesis (H0): μ = 9
# Alternative Hypothesis (H1): μ ≠ 9
print("\nClaim 1: The average daily working hours of warehouse employees is 9 hours.")
from scipy import stats
t_statistic, p_value = stats.ttest_1samp(sample, 9)
alpha = 0.05
print("t-statistic:", t_statistic)
print("p-value:", p_value)
if p_value < alpha:
    print("Reject the null hypothesis: The average daily working hours is significantly different from 9 hours.")
else:
    print("Fail to reject the null hypothesis: The average daily working hours is not significantly different from 9 hours.")


##Claim 2 (Overtime Issue):Less than 10% of employees are working more than 10 hours in a day.
# Null Hypothesis (H0): p >= 0.10
# Alternative Hypothesis (H1): p < 0.10
print("\nClaim 2 (Overtime Issue):Less than 10% of employees are working more than 10 hours in a day.")
overtime_count = np.sum(population > 10)
n = len(population)
p_hat = overtime_count / n
p0 = 0.10
z_statistic = (p_hat - p0) / np.sqrt((p0 * (1 - p0)) / n)
p_value = stats.norm.cdf(z_statistic)
alpha = 0.05
print("z-statistic:", z_statistic)
print("p-value:", p_value)
if p_value < alpha:
    print("Reject the null hypothesis: Less than 10% of employees are working more than 10 hours in a day.")
else:
    print("Fail to reject the null hypothesis: 10% or more employees are working more than 10 hours in a day.")

##Claim 3 (Underperformance Check):No employee works less than 8 hours in a day.
# Null Hypothesis (H0): p = 0
# Alternative Hypothesis (H1): p > 0
print("\nClaim 3 (Underperformance Check): No employee works less than 8 hours in a day.")
underperformance_count = np.sum(population < 8)
n = len(population)
p_hat = underperformance_count / n
p0 = 0
z_statistic = (p_hat - p0) / np.sqrt((p0 * (1 - p0)) / n) if p0*(1-p0) != 0 else 0
p_value = 1 - stats.norm.cdf(z_statistic)
alpha = 0.05
print("z-statistic:", z_statistic)
print("p-value:", p_value)
if p_value < alpha:
    print("Reject the null hypothesis: Some employees work less than 8 hours in a day.")
else:
    print("Fail to reject the null hypothesis: No employees work less than 8 hours in a day.")

## Claim 4 (Workload Balance):The working hours are evenly distributed without extreme outliers.
# Null Hypothesis (H0): No outliers present
# Alternative Hypothesis (H1): Outliers present
print("\nClaim 4 (Workload Balance): The working hours are evenly distributed without extreme outliers.")
Q1 = np.percentile(population, 25)
Q3 = np.percentile(population, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = population[(population < lower_bound) | (population > upper_bound)]
alpha = 0.05
if len(outliers) > 0:
    print("Reject the null hypothesis: Outliers are present in the working hours data.")
else:
    print("Fail to reject the null hypothesis: No outliers present in the working hours data.")

    print("No outliers detected in the working hours data.")

## Report
print("\nStatistical Analysis Report:")
print("1. Descriptive statistics indicate the central tendency and dispersion of working hours among employees.")
print("2. Hypothesis tests were conducted to evaluate claims regarding average working hours, overtime issues, underperformance, and workload balance.")
print("3. Visualizations such as histograms, boxplots, and Q-Q plots were used to assess the distribution and identify potential outliers in the data.")

