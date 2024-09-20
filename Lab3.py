import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc

data = {
    125: [2.7, 2.6, 4.6, 3.2, 3.0, 3.8],
    160: [4.6, 4.9, 5.0, 4.2, 3.6, 4.2],
    200: [4.6, 2.9, 3.4, 3.5, 4.1, 5.1]
}

df = pd.DataFrame(data)

# Perform ANOVA one way test
stat, p_value = stats.f_oneway(df[125], df[160], df[200])
print(stat, p_value)
print('stat=%.3f, p=%.3f' % (stat, p_value))

if p_value > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

# Reshape the DataFrame
df_melted = df.melt(var_name='Gas_Flow_Rate', value_name='Etch_Uniformity')

# Fit the ANOVA model
model = ols('Etch_Uniformity ~ C(Gas_Flow_Rate)', data=df_melted).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

comp = mc.MultiComparison(df_melted['Etch_Uniformity'], df_melted['Gas_Flow_Rate'])
post_hoc_res = comp.tukeyhsd()
print(post_hoc_res)

# Visualize the Tukey HSD results
comp.tukeyhsd().plot_simultaneous(xlabel='Etch Uniformity', ylabel='Gas Flow Rate')

# Create boxplot
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='Gas_Flow_Rate', y='Etch_Uniformity', data=df_melted, color='white')
ax = sns.swarmplot(x='Gas_Flow_Rate', y='Etch_Uniformity', data=df_melted, color='black')
plt.xlabel('Gas Flow Rate')
plt.ylabel('Etch Uniformity')
plt.title('Boxplot and Swarmplot of Etch Uniformity by Gas Flow Rate')
plt.show()

# Get predicted values and residuals
predicted = model.fittedvalues
residuals = model.resid

# res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
sm.qqplot(residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

# histogram
plt.hist(residuals, bins='auto', histtype='bar', ec='k')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Calculate means
mean_125 = df[125].mean()
mean_160 = df[160].mean()
mean_200 = df[200].mean()

print(f"Mean Etch Uniformity at 125 sccm: {mean_125:.2f}%")
print(f"Mean Etch Uniformity at 160 sccm: {mean_160:.2f}%")
print(f"Mean Etch Uniformity at 200 sccm: {mean_200:.2f}%")

def paired_t_test(sample1, sample2):
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)
    return t_statistic, p_value

t_statistic, p_value = paired_t_test(df[125], df[160])
print(f"t-statistic: {t_statistic:.3f}, p-value: {p_value:.3f}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the two groups.")
else:
    print("Fail to reject the null hypothesis: No significant difference between the two groups.")
t_statistic, p_value = paired_t_test(df[125], df[200])
print(f"t-statistic: {t_statistic:.3f}, p-value: {p_value:.3f}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the two groups.")
else:
    print("Fail to reject the null hypothesis: No significant difference between the two groups.")
t_statistic, p_value = paired_t_test(df[160], df[200])
print(f"t-statistic: {t_statistic:.3f}, p-value: {p_value:.3f}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the two groups.")
else:
    print("Fail to reject the null hypothesis: No significant difference between the two groups.")

# 4.53
mean = 30
sample_mean = 31.4
SE_mean = 0.336
Z = (sample_mean - mean) / SE_mean
print(f"Z-score: {Z:.2f}")
p_value = 2 * stats.norm.sf(abs(Z))
print(f"p-value: {p_value:.6f}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the two groups.")
else:
    print("Fail to reject the null hypothesis: No significant difference between the two groups.")

print("This is a two-sided test. The null hypothesis is H0: μ = 30, and the alternative hypothesis is H1: μ ≠ 30, which implies we are testing for a difference in both directions.")

print("(30.742,32.058)")
std_dev = 1.333
n = 15
SE = std_dev / np.sqrt(n)
print(f"Standard Error: {SE:.3f} std_dev / np.sqrt(n)")

# One sided P_value
# One sided P_value
p_value = stats.norm.sf(Z)
print(f"p-value: {p_value:.6f}")

