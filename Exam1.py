import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
from sklearn.linear_model import LinearRegression

## A bag contains 40 orange chips and 60 purple chips. Suppose 40 chips are randomly selected without watching with replacement (i.e. same color chip will be replaced once taken). What is the probability that 30 of the sampled chips will be purple? (Hint. The Excel function for combination is =Combin(N, k) and Python combination function is scipy.special.comb(N, k)) 
# chance of getting purple chip = 60/100 = 0.6
chance = scipy.special.comb(40, 30) * (0.6**30) * (0.4**10)
print(chance)

# Poisson distribution
sample_size = 10
prob = 0.05
lambda_ = sample_size * prob
poisson = stats.poisson.pmf(2, lambda_)
print(poisson)