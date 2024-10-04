import random
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from scipy.optimize import minimize

# Updated data with factors coded as 1 and -1
data = {
    'A': [-1.0, -1.0, -1.0, -1.0, -1.0,
          1.0, 1.0, 1.0, 1.0, 1.0,
          -1.0, -1.0, -1.0, -1.0, -1.0,
          1.0, 1.0, 1.0, 1.0, 1.0],
    
    'B': [-1.0, -1.0, -1.0, -1.0, -1.0,
          -1.0, -1.0, -1.0, -1.0, -1.0,
          1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0],
    
    'Hangtime': [1.68, 1.76, 1.71, 1.70, 1.84,
                 2.26, 2.16, 2.06, 2.35, 2.00,
                 1.76, 1.68, 1.88, 1.65, 1.79,
                 1.93, 1.95, 1.93, 1.91, 2.13]
}

# Create DataFrame
df = pd.DataFrame(data)

# Fit OLS model with numeric factors
model = ols('Hangtime ~ A + B + A:B', data=df).fit()
# Perform Type III ANOVA
aov_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA Table:")
print(aov_table)

print(model.summary())

# Define the function for Hangtime based on the regression coefficients
def hangtime_function(x):
    A, B = x  # Unpack A and B
    intercept = 1.9065
    beta_A = 0.1615
    beta_B = -0.0455
    beta_AB = -0.0525
    print(intercept + beta_A * A + beta_B * B + beta_AB * A * B)
    return -(intercept + beta_A * A + beta_B * B + beta_AB * A * B)  # Negative for maximization

# Bounds for A and B, they can vary between -1 and 1
bounds = [(-1, 1), (-1, 1)]

# Initial guess for A and B
x0 = [.5, -1]  # Start at A=0, B=0

# Perform the optimization
result = minimize(hangtime_function, x0, bounds=bounds)

# Since we minimized the negative of the Hangtime function, we negate the result to get the maximum Hangtime
optimal_A, optimal_B = result.x
max_hangtime = -result.fun


print(f"Optimal A: {optimal_A}")
print(f"Optimal B: {optimal_B}")
print(f"Maximum Hangtime: {max_hangtime}")

# normality test

import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate a Q-Q plot
residuals = model.resid
fig = sm.qqplot(residuals, line='s')
plt.show()

# Homogeneity of Variance check
# Levene's Test
# H0: All variances are equal

# Extract residuals for each factor level

stats.levene(df['Hangtime'][df['A'] == -1],
                df['Hangtime'][df['A'] == 1])

stats.levene(df['Hangtime'][df['B'] == -1],
                df['Hangtime'][df['B'] == 1])

print("Levene's Test for A:", stats.levene(df['Hangtime'][df['A'] == -1],
                df['Hangtime'][df['A'] == 1]))
print("Levene's Test for B:", stats.levene(df['Hangtime'][df['B'] == -1],
                df['Hangtime'][df['B'] == 1]))

# Constant variance assumption check

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

ax.set_title('box plot of Hangtime by A')
ax.set

data2 = [df['Hangtime'][df['A'] == -1], df['Hangtime'][df['A'] == 1]]

ax.boxplot(data2,
           labels= ['A= low', 'A=high'],
           showmeans= True)

plt.xlabel("A")
plt.ylabel("Hangtime")

plt.show()

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

ax.set_title('box plot of Hangtime by B')
ax.set

data2 = [df['Hangtime'][df['B'] == -1], df['Hangtime'][df['B'] == 1]]

ax.boxplot(data2,
              labels= ['B= low', 'B=high'],
              showmeans= True)

plt.xlabel("B")
plt.ylabel("Hangtime")

plt.show()


eij=model.resid

N = len(eij)
c = list(range(0, N))
order = random.sample(c, N) # this should be replaced by the exact run order if the random order is not used
print('order of observations/residuals is')
print(order)

eij=model.resid

x = range(0,N) # from the first collected obs to the last in order
y = eij[order] # assign run order (see the previous code box)

plt.scatter(x, y)
plt.xlabel('Order of observations')
plt.ylabel('Residuals')
plt.title('Residuals vs. Order of observations')
plt.show()
     