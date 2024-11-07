import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample structure of  dataset (fill this with collected data)
# replace the values once data is ready
data = {
    'Height_of_ramp': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    'Weight_of_car': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    'Size_of_front_wheels': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    'Size_of_back_wheels': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'Distance_traveled': [781.67, 890, 1123.33, 1015, 788.33, 1118.33, 1026.67, 991.67, 3593.33, 5110, 5010, 4820, 3756.67, 4620, 4683.33, 4366.67]  # Fill with data
}
df = pd.DataFrame(data)
data2=data
#Bultiple data points for each factor level for factorial design
Axel_distance = df['Distance_traveled'] * df['Height_of_ramp'] * df['Weight_of_car'] * df['Size_of_front_wheels'] * df['Size_of_back_wheels']
data2['Axel_distance']= Axel_distance
df = pd.DataFrame(data2)

# ANOVA Analysis
model = ols('Distance_traveled ~ Height_of_ramp + Weight_of_car + Size_of_front_wheels + Size_of_back_wheels + ABCD', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("Model Summary:")
print(model.summary())

print("ANOVA Table:")
print(anova_table)

# Regression Modeling
X = df[['Height_of_ramp', 'Weight_of_car', 'Size_of_front_wheels', 'Size_of_back_wheels', 'ABCD']]
y = df['Distance_traveled']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predict on the test set
y_pred = reg_model.predict(X_test)

# Calculate and print model metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nRegression Model Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot residuals
sns.residplot(x=y_pred, y=y_test - y_pred, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()