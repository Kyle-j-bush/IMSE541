import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.optimize import minimize

# Data for factorial design
data = {
    'a': [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
    'b': [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
    'c': [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1], 
    'Response': [221, 325, 354, 552, 440, 406, 605, 392, 311, 435, 348, 472, 453, 377, 500, 419] 
}

# Create a DataFrame
df = pd.DataFrame(data)

# Fit the model using OLS regression
model = ols('Response ~ a + b + a:b + c + a:c + b:c + a:b:c', data=df).fit()

# Display ANOVA table
aov_table = sm.stats.anova_lm(model, typ=3)
print("ANOVA Table:")
print(aov_table)

# Gigen the output we can conclude that b and c are both significant factors as well as the interaction between a and c being the most significant

# Display model summary
print(model.summary())

# Define the tool life function
def ToolLife_function(x):
    a, b, c = x
    intercept = 413.125
    beta_a = 9.1215
    beta_b = 42.125
    beta_c = 35.875
    beta_ab = -5.625
    beta_ac = -59.625
    beta_bc = -12.125
    beta_abc = 17.375
    return -(intercept + beta_a * a + beta_b * b + beta_c * c + beta_ab * a * b + beta_ac * a * c + beta_bc * b * c + beta_abc * a * b * c)

# Define bounds for optimization (between -1 and 1 for each factor)
bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

# Initial guess for factors
x0 = [0.0, 0.0, 0.0]  # Start at center point

# Perform optimization to maximize tool life
result = minimize(ToolLife_function, x0, bounds=bounds)

# Extract optimal values and maximum tool life
optimal_a, optimal_b, optimal_c = result.x
max_tool_life = -result.fun  # Negate to get the maximum value

# Print results
print(f"Optimal a: {optimal_a}")
print(f"Optimal b: {optimal_b}")
print(f"Optimal c: {optimal_c}")
print(f"Maximum tool life: {max_tool_life}")
# see printed results for answers

# Define bounds for optimization (between -1 and 1 for each factor)
bounds = [(-1.0, 1.0), (0,0), (-1.0, 1.0)]

# Initial guess for factors
x0 = [0.0, 0.0, 0.0]  # Start at center point

# Perform optimization to maximize tool life
result = minimize(ToolLife_function, x0, bounds=bounds)

# Extract optimal values and maximum tool life
optimal_a, optimal_b, optimal_c = result.x
max_tool_life = -result.fun  # Negate to get the maximum value

# Print results
print(f"Optimal a: {optimal_a}")
print(f"Optimal b: {optimal_b}")
print(f"Optimal c: {optimal_c}")
print(f"Maximum tool life With Fixed B: {max_tool_life}")
# Because the interaction of b is not significant when looking at a:b, b:c, and a:b:c we can conclude B does not play a significant role in the model
# This is also supported by the fact that when b is fixed at 0 the model the tool life does not suffer a significant decrease

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.optimize import minimize

# Data for factorial design
data = {
    'a': [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],  
    'b': [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],  
    'c': [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1], 
    'd': [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1],  
    'Response': [188, 172, 179, 185, 175, 183, 190, 175, 200, 170, 189, 183, 201, 181, 189, 178]  
}

# Create a DataFrame
df = pd.DataFrame(data)

# Fit the model using OLS regression
model = ols('Response ~ a + b + c + d + a:b + a:c + a:d + b:c + b:d + c:d + a:b:c + a:b:d + a:c:d + b:c:d + a:b:c:d', data=df).fit()

# Display ANOVA table
aov_table = sm.stats.anova_lm(model)
print("ANOVA Table:")
print(aov_table)

# Display model summary
print(model.summary())


# Define the response function for tool life
def tool_life_function(x):
    a, b, c, d = x
    intercept = 183.6250
    beta_a = -5.2500
    beta_b = -0.1250
    beta_c = 0.3750
    beta_d = 2.7500
    beta_ab = 2.0000
    beta_ac = 0.5000
    beta_ad = -3.1250
    beta_bc = -0.8750
    beta_bd = -1.5000
    beta_cd = 0.5000
    beta_abc = -3.7500
    beta_abd = -2.1250
    beta_acd = 0.1250
    beta_bcd = -1.2500
    beta_abcd = 1.8750
    return -(intercept + beta_a * a + beta_b * b + beta_c * c + beta_d * d +
             beta_ab * a * b + beta_ac * a * c + beta_ad * a * d +
             beta_bc * b * c + beta_bd * b * d + beta_cd * c * d +
             beta_abc * a * b * c + beta_abd * a * b * d +
             beta_acd * a * c * d + beta_bcd * b * c * d + beta_abcd * a * b * c * d)

# Define bounds for optimization (between -1 and 1 for each factor)
bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

# Initial guess for factors
x0 = [0.0, 0.0, 0.0, 0.0]  

# Perform optimization to maximize tool life
result = minimize(tool_life_function, x0, bounds=bounds)

# Extract optimal values and maximum tool life
optimal_a, optimal_b, optimal_c, optimal_d = result.x
max_tool_life = -result.fun  # Negate to get the maximum value