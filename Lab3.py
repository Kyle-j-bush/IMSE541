# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
from sklearn.linear_model import LinearRegression

## region 4.35

data = {
    125: [2.7, 2.6, 4.6, 3.2, 3.0, 3.8],
    160: [4.6, 4.9, 5.0, 4.2, 3.6, 4.2],
    200: [4.6, 2.9, 3.4, 3.5, 4.1, 5.1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# region 4.35 (a)

# Perform ANOVA one way test
stat, p_value = stats.f_oneway(df[125], df[160], df[200])

# Print the results
print(f"ANOVA one-way test results:\nStatistic: {stat:.3f}, p-value: {p_value:.3f}")

# Interpret the results
if p_value > 0.05:
    print('Interpretation: Probably the same distribution')
else:
    print('Interpretation: Probably different distributions')

# Reshape the DataFrame
df_melted = df.melt(var_name='Gas_Flow_Rate', value_name='Etch_Uniformity')

# Fit the ANOVA model
model = ols('Etch_Uniformity ~ C(Gas_Flow_Rate)', data=df_melted).fit()
# Create the ANOVA table
anova_table = sm.stats.anova_lm(model, typ=2)
# Print the ANOVA table
print("\nANOVA Table:")
print(anova_table)

# Get the residuals
comp = mc.MultiComparison(df_melted['Etch_Uniformity'], df_melted['Gas_Flow_Rate'])
# Perform the Tukey HSD test
post_hoc_res = comp.tukeyhsd()
# Print the Tukey HSD results
print("\nTukey HSD Test Results:")
print(post_hoc_res)

# Visualize the Tukey HSD results
comp.tukeyhsd().plot_simultaneous(xlabel='Etch Uniformity', ylabel='Gas Flow Rate')
# endregion

# region 4.35 (b)
# Create boxplot
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='Gas_Flow_Rate', y='Etch_Uniformity', data=df_melted, color='white')
ax = sns.swarmplot(x='Gas_Flow_Rate', y='Etch_Uniformity', data=df_melted, color='black')
plt.xlabel('Gas Flow Rate')
plt.ylabel('Etch Uniformity')
plt.title('Boxplot and Swarmplot of Etch Uniformity by Gas Flow Rate')
plt.show()

print("\nConclusion: Because 125 has the lowest mean, it is the best gas flow rate to use.")
# endregion

# region 4.35 (c)
# Get predicted values and residuals
predicted = model.fittedvalues
residuals = model.resid

# Q-Q plot
sm.qqplot(residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()
print("Q-Q Plot Interpretation: The residuals are close to the line, indicating that the residuals are normally distributed.")
print("The normality assumption does seem reasonable.\n")
# endregion

# region 4.35 (d)
# Perform the Shapiro-Wilk test
stat, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test:\nStatistic: {stat:.3f}, p-value: {p_value:.3f}")
alpha = 0.05
if p_value > alpha:
    print("Interpretation: Fail to reject the null hypothesis - The residuals are normally distributed.\n")
else:
    print("Interpretation: Reject the null hypothesis - The residuals are not normally distributed.\n")
# endregion

# endregion

## region 4.56
# Calculate means
mean_125 = df[125].mean()
mean_160 = df[160].mean()
mean_200 = df[200].mean()

print(f"Mean Etch Uniformity:\n125 sccm: {mean_125:.2f}%\n160 sccm: {mean_160:.2f}%\n200 sccm: {mean_200:.2f}%\n")

# Function to perform a paired t-test
def paired_t_test(sample1, sample2):
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)
    return t_statistic, p_value

# Perform paired t-tests
t_statistic, p_value = paired_t_test(df[125], df[160])
print(f"Paired t-test between 125 and 160:\nt-statistic: {t_statistic:.3f}, p-value: {p_value:.3f}")
if p_value < alpha:
    print("Interpretation: Reject the null hypothesis - Significant difference between the two groups.\n")
else:
    print("Interpretation: Fail to reject the null hypothesis - No significant difference between the two groups.\n")

t_statistic, p_value = paired_t_test(df[125], df[200])
print(f"Paired t-test between 125 and 200:\nt-statistic: {t_statistic:.3f}, p-value: {p_value:.3f}")
if p_value < alpha:
    print("Interpretation: Reject the null hypothesis - Significant difference between the two groups.\n")
else:
    print("Interpretation: Fail to reject the null hypothesis - No significant difference between the two groups.\n")

t_statistic, p_value = paired_t_test(df[160], df[200])
print(f"Paired t-test between 160 and 200:\nt-statistic: {t_statistic:.3f}, p-value: {p_value:.3f}")
if p_value < alpha:
    print("Interpretation: Reject the null hypothesis - Significant difference between the two groups.\n")
else:
    print("Interpretation: Fail to reject the null hypothesis - No significant difference between the two groups.\n")
# endregion

## region 4.53

# region 4.53 (a)
# Calculate the mean, standard deviation, and standard error of the mean
mean = 30
sample_mean = 31.4
SE_mean = 0.336

# Calculate the Z-score and p-value
Z = (sample_mean - mean) / SE_mean
print(f"Z-score: {Z:.2f}")
p_value = 2 * stats.norm.sf(abs(Z))

# Print the results and interpret the p-value
print(f"p-value: {p_value:.6f}")
if p_value < alpha:
    print("Interpretation: Reject the null hypothesis - Significant difference between the two groups.\n")
else:
    print("Interpretation: Fail to reject the null hypothesis - No significant difference between the two groups.\n")
# endregion

# region 4.53 (b)
print("Two-sided test:\nThe null hypothesis is H0: μ = 30, and the alternative hypothesis is H1: μ ≠ 30, which implies we are testing for a difference in both directions.\n")
# endregion

# region 4.53 (c)
print("Confidence Interval: (30.742, 32.058)\n")
# endregion

# region 4.53 (d)
std_dev = 1.333
n = 15
SE = std_dev / np.sqrt(n)
print(f"Standard Error: {SE:.3f}\n")
# endregion

# region 4.53 (e)
# One-sided p-value
p_value = stats.norm.sf(Z)
print(f"One-sided p-value: {p_value:.6f}\n")
# endregion

# endregion

## region 4.47

# Create a DataFrame
data = {
    "Brake Horsepower": [225, 212, 229, 222, 219, 278, 246, 237, 233, 224, 223, 230],
    "rpm": [2000, 1800, 2400, 1900, 1600, 2500, 3000, 3200, 2800, 3400, 1800, 2500],
    "Road Octane Number": [90, 94, 88, 91, 86, 96, 94, 90, 88, 86, 90, 89],
    "Compression": [100, 95, 110, 96, 100, 110, 98, 100, 105, 97, 100, 104]
}

df = pd.DataFrame(data)

# region 4.47 (a)
# Fit the linear regression model
model = LinearRegression()
model.fit(df[['rpm', 'Road Octane Number', 'Compression']], df['Brake Horsepower'])

# Get the coefficients
coefficients = model.coef_
intercept = model.intercept_
print(f"Linear Regression Coefficients:\n{coefficients}")
print(f"Intercept: {intercept}")

# Predict new values
predicted = model.predict(df[['rpm', 'Road Octane Number', 'Compression']])

# Print the predicted values
print(f"Predicted Brake Horsepower: {predicted}\n")
# endregion

# region 4.47 (b)
# Test for significance of regression
X = df[['rpm', 'Road Octane Number', 'Compression']]
y = df['Brake Horsepower']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print("Regression Model Summary:")
print(model.summary())
# endregion

# region 4.47 (c)
print("Significance Interpretation: Only rpm is significant in predicting Brake Horsepower.\n")
# endregion

# endregion
