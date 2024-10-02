import math
import random
import scipy
import statistics
import numpy as np
import scipy.stats
from scipy import stats
from scipy.stats import f_oneway
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc

# Updated Data dictionary to include 'Vibration'
#B is width of support
#C is number of paperclips
Data = {
    'A': [-1, 1, -1, -1, 1, 1, -1, 1],
    'B': [-1, 1, 1, -1, 1, -1, 1, 1],
    'C': [-1, 1, -1, 1, -1, 1, 1, 1],
    'Hangtime': []
}

# Create DataFrame with the updated columns
df = pd.DataFrame(Data, columns=['A', 'B', 'C', 'Hangtime'])

# Updated formula to include 'C' and interactions
modelA = ols('Hangtime ~ A + B + C + A:B + A:C + B:C + A:B:C', data=df).fit()
aov_tableA = sm.stats.anova_lm(modelA, typ=2)

#Residual Analysis
residuals = modelA.resid
print(residuals)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(residuals, plot=plt, rvalue=True)
ax.set_title("Normal Probability Plot of Residuals")
ax.set_xlabel("Theoretical Quantiles")

plt.show()

# Homogeneity of Variance check
# if the p-value is small, reject H0 that the variances are equal
# if the p-value is large, fail to reject H0 that the variances are equal

levene_A = stats.levene(df['Hangtime'][df['A'] == 1],
                        df['Hangtime'][df['A'] == -1],)

levene_B = stats.levene(df['Hangtime'][df['B'] == 1],
                        df['Hangtime'][df['B'] == -1])

levene_C = stats.levene(df['Hangtime'][df['C'] == 1],
                        df['Hangtime'][df['C'] == -1])

print(levene_A)
print(levene_B)
print(levene_C)

# Box plots by factor Primer_type'
# If the IRQ=75% quartitle - 25% quartile are very different, then question the constant variance assumption
# If the F test p value is very small, we reject H0 and can use this box plot to identify the best factor level

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

ax.set_title("Box Plot of Hangtime by A - Length of Blade", fontsize=20)
ax.set 

data2 = [df['Hangtime'][df['A'] == 1],
         df['Hangtime'][df['A'] == -1]]

ax.boxplot(data2, labels=['A=1', 'A=-1'], showmeans = True)

plt.xlabel("A - Length of Blade", fontsize=20)
plt.ylabel("Hangtime", fontsize=20)

plt.show()

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

ax.set_title("Box Plot of Hangtime by B - Width Of Support", fontsize=20)
ax.set 

data2 = [df['Hangtime'][df['B'] == 1],
         df['Hangtime'][df['B'] == -1]]

ax.boxplot(data2, labels=['B=1', 'B=-1'], showmeans = True)

plt.xlabel("B - Width of Support", fontsize=20)
plt.ylabel("Hangtime", fontsize=20)

plt.show()

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

ax.set_title("Box Plot of Hangtime by C - Number of Paperclips", fontsize=20)
ax.set

data2 = [df['Hangtime'][df['C'] == 1],
            df['Hangtime'][df['C'] == -1]]

ax.boxplot(data2, labels=['C=1', 'C=-1'], showmeans = True)

plt.xlabel("C - Number of Paperclips", fontsize=20)
plt.ylabel("Hangtime", fontsize=20)

plt.show()


# testing the random order for collecting observations
# Minitab calls the order variable, run order
# the following code generate a random order
# you should conduct your experiments using the random order generated
# for example, the first number is 5, it means that you will run the setting 5 in the experimentation table
# observation label 5 is primier type 3 and application method 2 (spraying)
# the run observation 11 and so on until all observations are collected
# import random (see the first cell)
N = len(residuals)
c = list(range(0, N))
order = random.sample(c, N)
print(order)