#13.22 a.) The Generator of this experiment is variable E: Delay
#13.22 b.) The reslution of this design is 2^5-1 meaning the resolution is V with 16 runs needed

import pandas as pd
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# Data from Table 13E.7
data = {
    'A': [-1, -1, -1, -1, +1, +1, +1, +1, -1, -1, -1, -1, +1, +1, +1, +1],
    'B': [-1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1],
    'C': [-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1],
    'D': [-1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1],
    'E': [-1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1],
    'Std_Dev': [1.13, 1.25, 0.97, 1.70, 1.47, 1.28, 1.18, 0.98, 
                0.78, 1.36, 1.85, 0.62, 1.09, 1.10, 0.76, 2.10]
}
# Create DataFrame
df = pd.DataFrame(data)

# Fit the model using OLS regression with main effects and all interactions
model = ols('Std_Dev ~ A + B + C + D + E', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ = 2)  # Type II ANOVA Table

print(anova_table)
print(model.summary())

# no effect is very large: d is the factor with the largest effect


# Q-Q Plot to check for normality of residuals
sm.qqplot(model.resid, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()

# Residuals vs. Fitted Values Plot
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()

# The Q-Q plot shows that the residuals are not normally distributed, and the residuals vs. fitted values plot shows that the residuals are not homoscedastic. 
# This indicates that the model assumptions are violated.

# Because the residuals are not normally distributed and the residuals vs. fitted values plot shows heteroscedasticity, the model assumptions are violated. 
# This means that the model may not be appropriate for the data, and the results and conclusions drawn from the model may not be reliable. Additional diagnostics and 
# model adjustments may be needed to address these issues.