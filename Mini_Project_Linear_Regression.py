#!/usr/bin/env python
# coding: utf-8

# # Regression in Python
# 
# ***
# This is a very quick run-through of some basic statistical concepts, adapted from [Lab 4 in Harvard's CS109](https://github.com/cs109/2015lab4) course. Please feel free to try the original lab if you're feeling ambitious :-) The CS109 git repository also has the solutions if you're stuck.
# 
# * Linear Regression Models
# * Prediction using linear regression
# 
# Linear regression is used to model and predict continuous outcomes with normal random errors. There are nearly an infinite number of different types of regression models and each regression model is typically defined by the distribution of the prediction errors (called "residuals") of the type of data. Logistic regression is used to model binary outcomes whereas Poisson regression is used to predict counts. In this exercise, we'll see some examples of linear regression as well as Train-test splits.
# 
# The packages we'll cover are: `statsmodels`, `seaborn`, and `scikit-learn`. While we don't explicitly teach `statsmodels` and `seaborn` in the Springboard workshop, those are great libraries to know.
# ***

# <img width=600 height=300 src="https://imgs.xkcd.com/comics/sustainable.png"/>
# ***

# In[1]:


# special IPython command to prepare the notebook for matplotlib and other libraries
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

import seaborn as sns

# special matplotlib argument for improved plots
from matplotlib import rcParams
sns.set_style("whitegrid")
sns.set_context("poster")


# ***
# # Part 1: Introduction to Linear Regression
# ### Purpose of linear regression
# ***
# <div class="span5 alert alert-info">
# 
# <p> Given a dataset containing predictor variables $X$ and outcome/response variable $Y$, linear regression can be used to: </p>
# <ul>
#   <li> Build a <b>predictive model</b> to predict future values of $\hat{Y}$, using new data $X^*$ where $Y$ is unknown.</li>
#   <li> Model the <b>strength of the relationship</b> between each independent variable $X_i$ and $Y$</li>
#     <ul>
#       <li> Many times, only a subset of independent variables $X_i$ will have a linear relationship with $Y$</li>
#       <li> Need to figure out which $X_i$ contributes most information to predict $Y$ </li>
#     </ul>
#    <li>It is in many cases, the first pass prediction algorithm for continuous outcomes. </li>
# </ul>
# </div>
# 
# ### A Brief Mathematical Recap
# ***
# 
# [Linear Regression](http://en.wikipedia.org/wiki/Linear_regression) is a method to model the relationship between a set of independent variables $X$ (also knowns as explanatory variables, features, predictors) and a dependent variable $Y$.  This method assumes the relationship between each predictor $X$ is **linearly** related to the dependent variable $Y$. The most basic linear regression model contains one independent variable $X$, we'll call this the simple model. 
# 
# $$ Y = \beta_0 + \beta_1 X + \epsilon$$
# 
# where $\epsilon$ is considered as an unobservable random variable that adds noise to the linear relationship. In linear regression, $\epsilon$ is assumed to be normally distributed with a mean of 0. In other words, what this means is that on average, if we know $Y$, a roughly equal number of predictions $\hat{Y}$ will be above $Y$ and others will be below $Y$. That is, on average, the error is zero. The residuals, $\epsilon$ are also assumed to be "i.i.d.": independently and identically distributed. Independence means that the residuals are not correlated -- the residual from one prediction has no effect on the residual from another prediction. Correlated errors are common in time series analysis and spatial analyses.
# 
# * $\beta_0$ is the intercept of the linear model and represents the average of $Y$ when all independent variables $X$ are set to 0.
# 
# * $\beta_1$ is the slope of the line associated with the regression model and represents the average effect of a one-unit increase in $X$ on $Y$.
# 
# * Back to the simple model. The model in linear regression is the *conditional mean* of $Y$ given the values in $X$ is expressed a linear function.  
# 
# $$ y = f(x) = E(Y | X = x)$$ 
# 
# ![conditional mean](images/conditionalmean.png)
# http://www.learner.org/courses/againstallodds/about/glossary.html
# 
# * The goal is to estimate the coefficients (e.g. $\beta_0$ and $\beta_1$). We represent the estimates of the coefficients with a "hat" on top of the letter.  
# 
# $$ \hat{\beta}_0, \hat{\beta}_1 $$
# 
# * Once we estimate the coefficients $\hat{\beta}_0$ and $\hat{\beta}_1$, we can use these to predict new values of $Y$ given new data $X$.
# 
# $$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1$$
# 
# * Multiple linear regression is when you have more than one independent variable and the estimation involves matrices
#     * $X_1$, $X_2$, $X_3$, $\ldots$
# 
# 
# * How do you estimate the coefficients? 
#     * There are many ways to fit a linear regression model
#     * The method called **least squares** is the most common methods
#     * We will discuss least squares
# 
# $$ Y = \beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p + \epsilon$$ 
#     
# ### Estimating $\hat\beta$: Least squares
# ***
# [Least squares](http://en.wikipedia.org/wiki/Least_squares) is a method that can estimate the coefficients of a linear model by minimizing the squared residuals: 
# 
# $$ \mathscr{L} = \sum_{i=1}^N \epsilon_i^2 = \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2  = \sum_{i=1}^N \left(y_i - \left(\beta_0 + \beta_1 x_i\right)\right)^2 $$
# 
# where $N$ is the number of observations and $\epsilon$ represents a residual or error, ACTUAL - PREDICTED.  
# 
# #### Estimating the intercept $\hat{\beta_0}$ for the simple linear model
# 
# We want to minimize the squared residuals and solve for $\hat{\beta_0}$ so we take the partial derivative of $\mathscr{L}$ with respect to $\hat{\beta_0}$ 

# $
# \begin{align}
# \frac{\partial \mathscr{L}}{\partial \hat{\beta_0}} &= \frac{\partial}{\partial \hat{\beta_0}} \sum_{i=1}^N \epsilon^2 \\
# &= \frac{\partial}{\partial \hat{\beta_0}} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2 \\
# &= \frac{\partial}{\partial \hat{\beta_0}} \sum_{i=1}^N \left( y_i - \left( \hat{\beta}_0 + \hat{\beta}_1 x_i \right) \right)^2 \\
# &= -2 \sum_{i=1}^N \left( y_i - \left( \hat{\beta}_0 + \hat{\beta}_1 x_i \right) \right) \hspace{25mm} \mbox{(by chain rule)} \\
# &= -2 \sum_{i=1}^N (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) \\
# &= -2 \left[ \left( \sum_{i=1}^N y_i \right) - N \hat{\beta_0} - \hat{\beta}_1 \left( \sum_{i=1}^N x_i
# \right) \right] \\
# & 2 \left[ N \hat{\beta}_0 + \hat{\beta}_1 \sum_{i=1}^N x_i - \sum_{i=1}^N y_i \right] = 0 \hspace{20mm} \mbox{(Set equal to 0 and solve for $\hat{\beta}_0$)} \\
# & N \hat{\beta}_0 + \hat{\beta}_1 \sum_{i=1}^N x_i - \sum_{i=1}^N y_i = 0 \\
# & N \hat{\beta}_0 = \sum_{i=1}^N y_i - \hat{\beta}_1 \sum_{i=1}^N x_i \\
# & \hat{\beta}_0 = \frac{\sum_{i=1}^N y_i - \hat{\beta}_1 \sum_{i=1}^N x_i}{N} \\
# & \hat{\beta}_0 = \frac{\sum_{i=1}^N y_i}{N} - \hat{\beta}_1 \frac{\sum_{i=1}^N x_i}{N} \\
# & \boxed{\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}}
# \end{align}
# $

# Using this new information, we can compute the estimate for $\hat{\beta}_1$ by taking the partial derivative of $\mathscr{L}$ with respect to $\hat{\beta}_1$.

# $
# \begin{align}
# \frac{\partial \mathscr{L}}{\partial \hat{\beta_1}} &= \frac{\partial}{\partial \hat{\beta_1}} \sum_{i=1}^N \epsilon^2 \\
# &= \frac{\partial}{\partial \hat{\beta_1}} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2 \\
# &= \frac{\partial}{\partial \hat{\beta_1}} \sum_{i=1}^N \left( y_i - \left( \hat{\beta}_0 + \hat{\beta}_1 x_i \right) \right)^2 \\
# &= 2 \sum_{i=1}^N \left( y_i - \left( \hat{\beta}_0 + \hat{\beta}_1 x_i \right) \right) \left( -x_i \right) \hspace{25mm}\mbox{(by chain rule)} \\
# &= -2 \sum_{i=1}^N x_i \left( y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i \right) \\
# &= -2 \sum_{i=1}^N x_i (y_i - \hat{\beta}_0 x_i - \hat{\beta}_1 x_i^2) \\
# &= -2 \sum_{i=1}^N x_i (y_i - \left( \bar{y} - \hat{\beta}_1 \bar{x} \right) x_i - \hat{\beta}_1 x_i^2) \\
# &= -2 \sum_{i=1}^N (x_i y_i - \bar{y}x_i + \hat{\beta}_1\bar{x}x_i - \hat{\beta}_1 x_i^2) \\
# &= -2 \left[ \sum_{i=1}^N x_i y_i - \bar{y} \sum_{i=1}^N x_i + \hat{\beta}_1\bar{x}\sum_{i=1}^N x_i - \hat{\beta}_1 \sum_{i=1}^N x_i^2 \right] \\
# &= -2 \left[ \hat{\beta}_1 \left\{ \bar{x} \sum_{i=1}^N x_i - \sum_{i=1}^N x_i^2 \right\} + \left\{ \sum_{i=1}^N x_i y_i - \bar{y} \sum_{i=1}^N x_i \right\}\right] \\
# & 2 \left[ \hat{\beta}_1 \left\{ \sum_{i=1}^N x_i^2 - \bar{x} \sum_{i=1}^N x_i \right\} + \left\{ \bar{y} \sum_{i=1}^N x_i - \sum_{i=1}^N x_i y_i \right\} \right] = 0 \\
# & \hat{\beta}_1 = \frac{-\left( \bar{y} \sum_{i=1}^N x_i - \sum_{i=1}^N x_i y_i \right)}{\sum_{i=1}^N x_i^2 - \bar{x}\sum_{i=1}^N x_i} \\
# &= \frac{\sum_{i=1}^N x_i y_i - \bar{y} \sum_{i=1}^N x_i}{\sum_{i=1}^N x_i^2 - \bar{x} \sum_{i=1}^N x_i} \\
# & \boxed{\hat{\beta}_1 = \frac{\sum_{i=1}^N x_i y_i - \bar{x}\bar{y}n}{\sum_{i=1}^N x_i^2 - n \bar{x}^2}}
# \end{align}
# $

# The solution can be written in compact matrix notation as
# 
# $$\hat\beta =  (X^T X)^{-1}X^T Y$$ 
# 
# We wanted to show you this in case you remember linear algebra, in order for this solution to exist we need $X^T X$ to be invertible. Of course this requires a few extra assumptions, $X$ must be full rank so that $X^T X$ is invertible, etc. Basically, $X^T X$ is full rank if all rows and columns are linearly independent. This has a loose relationship to variables and observations being independent respective. **This is important for us because this means that having redundant features in our regression models will lead to poorly fitting (and unstable) models.** We'll see an implementation of this in the extra linear regression example.

# ***
# # Part 2: Exploratory Data Analysis for Linear Relationships
# 
# The [Boston Housing data set](https://archive.ics.uci.edu/ml/datasets/Housing) contains information about the housing values in suburbs of Boston.  This dataset was originally taken from the StatLib library which is maintained at Carnegie Mellon University and is now available on the UCI Machine Learning Repository. 
# 
# 
# ## Load the Boston Housing data set from `sklearn`
# ***
# 
# This data set is available in the [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston) python module which is how we will access it today.  

# In[2]:


from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()


# In[3]:


boston.keys()


# In[4]:


boston.data.shape


# In[5]:


# Print column names
print(boston.feature_names)


# In[6]:


# Print description of Boston housing data set
print(boston.DESCR)


# Now let's explore the data set itself. 

# In[7]:


bos = pd.DataFrame(boston.data)
bos.head()


# There are no column names in the DataFrame. Let's add those. 

# In[8]:


bos.columns = boston.feature_names
bos.head()


# Now we have a pandas DataFrame called `bos` containing all the data we want to use to predict Boston Housing prices.  Let's create a variable called `PRICE` which will contain the prices. This information is contained in the `target` data. 

# In[9]:


print(boston.target.shape)


# In[10]:


bos['PRICE'] = boston.target
bos.head()


# ## EDA and Summary Statistics
# ***
# 
# Let's explore this data set.  First we use `describe()` to get basic summary statistics for each of the columns. 

# In[11]:


bos.describe()


# ### Scatterplots
# ***
# 
# Let's look at some scatter plots for three variables: 'CRIM' (per capita crime rate), 'RM' (number of rooms) and 'PTRATIO' (pupil-to-teacher ratio in schools).  

# In[12]:


plt.scatter(bos.CRIM, bos.PRICE)
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")


# <div class="span5 alert alert-info">
# <h3>Part 2 Checkup Exercise Set I</h3>
# 
# <p><b>Exercise:</b> What kind of relationship do you see? e.g. positive, negative?  linear? non-linear? Is there anything else strange or interesting about the data? What about outliers?</p>
# 
# 
# <p><b>Exercise:</b> Create scatter plots between *RM* and *PRICE*, and *PTRATIO* and *PRICE*. Label your axes appropriately using human readable labels. Tell a story about what you see.</p>
# 
# <p><b>Exercise:</b> What are some other numeric variables of interest? Why do you think they are interesting? Plot scatterplots with these variables and *PRICE* (house price) and tell a story about what you see.</p>
# 
# </div>

# In[22]:


# your turn: describe relationship
# CRIM and Price seems seems to be negatively coorelated.LSTAT has a high negative corelation with Price and RM has a high positive coorelation. 
#Also variables like RAD and TAX are highly coorelated, so is AGE and DIX , Nox and AGE etc.
#
plt.figure(figsize=(20,10))
correlation_matrix = bos.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True,cmap="YlGnBu")
plt.show()


# In[19]:


# your turn: scatter plot between *RM* and *PRICE*
plt.figure(figsize=(20,10))
plt.scatter(bos.RM,bos.PRICE)
plt.xlabel("Avg rooms per dwelling")
plt.title("Relationships between avg rooms per dwelling and price")
plt.show()


# In[18]:


# your turn: scatter plot between *PTRATIO* and *PRICE*
plt.figure(figsize=(20,10))
plt.scatter(bos.PTRATIO,bos.PRICE)
plt.xlabel("Pupil teacher ratio by town")
plt.title("Relationships between pupil teacher ratio by town and price")
plt.show()


# ### Scatterplots using Seaborn
# ***
# 
# [Seaborn](https://stanford.edu/~mwaskom/software/seaborn/) is a cool Python plotting library built on top of matplotlib. It provides convenient syntax and shortcuts for many common types of plots, along with better-looking defaults.
# 
# We can also use [seaborn regplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/regression.html#functions-to-draw-linear-regression-models) for the scatterplot above. This provides automatic linear regression fits (useful for data exploration later on). Here's one example below.

# In[23]:


sns.regplot(y="PRICE", x="RM", data=bos, fit_reg = True)


# ### Histograms
# ***
# 

# In[25]:


plt.hist(np.log(bos.CRIM))
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequency")
plt.show()


# <div class="span5 alert alert-info">
# <h3>Part 2 Checkup Exercise Set II</h3>
# 
# <p><b>Exercise:</b> In the above histogram, we took the logarithm of the crime rate per capita. Repeat this histogram without taking the log. What was the purpose of taking the log? What do we gain by making this transformation? What do you now notice about this variable that is not obvious without making the transformation?
# 
# <p><b>Exercise:</b> Plot the histogram for *RM* and *PTRATIO* against each other, along with the two variables you picked in the previous section. We are looking for correlations in predictors here.</p>
# </div>

# In[27]:


#your turn
#logarithmic transformations make the distribution more normalized which helps in identifying how skewed your data was previously. 

plt.hist(bos.CRIM)
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequency")
plt.show()

plt.hist(bos.RM)
plt.title("# of Rooms")
plt.xlabel("# of Rooms")
plt.ylabel("Frequency")
plt.show()

plt.hist(bos.PTRATIO)
plt.title("Pupil to Teacher Ratio")
plt.xlabel("Pupil to Teacher Ratio")
plt.ylabel("Frequency")
plt.show()

plt.hist(bos.NOX)
plt.title("NOx Concentration")
plt.xlabel("NOx Concentration (pp10m)")
plt.ylabel("Frequency")
plt.show()

plt.hist(bos.TAX)
plt.title("Tax rate per $10,000")
plt.xlabel("Tax Rate per $10,000")
plt.ylabel("Frequency")
plt.show()


# ## Part 3: Linear Regression with Boston Housing Data Example
# ***
# 
# Here, 
# 
# $Y$ = boston housing prices (called "target" data in python, and referred to as the dependent variable or response variable)
# 
# and
# 
# $X$ = all the other features (or independent variables, predictors or explanatory variables)
# 
# which we will use to fit a linear regression model and predict Boston housing prices. We will use the least-squares method to estimate the coefficients.  

# We'll use two ways of fitting a linear regression. We recommend the first but the second is also powerful in its features.

# ### Fitting Linear Regression using `statsmodels`
# ***
# [Statsmodels](http://statsmodels.sourceforge.net/) is a great Python library for a lot of basic and inferential statistics. It also provides basic regression functions using an R-like syntax, so it's commonly used by statisticians. While we don't cover statsmodels officially in the Data Science Intensive workshop, it's a good library to have in your toolbox. Here's a quick example of what you could do with it. The version of least-squares we will use in statsmodels is called *ordinary least-squares (OLS)*. There are many other versions of least-squares such as [partial least squares (PLS)](https://en.wikipedia.org/wiki/Partial_least_squares_regression) and [weighted least squares (WLS)](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares).

# In[28]:


# Import regression modules
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[29]:


# statsmodels works nicely with pandas dataframes
# The thing inside the "quotes" is called a formula, a bit on that below
m = ols('PRICE ~ RM',bos).fit()
print(m.summary())


# #### Interpreting coefficients
# 
# There is a ton of information in this output. But we'll concentrate on the coefficient table (middle table). We can interpret the `RM` coefficient (9.1021) by first noticing that the p-value (under `P>|t|`) is so small, basically zero. This means that the number of rooms, `RM`, is a statisticall significant predictor of `PRICE`. The regression coefficient for `RM` of 9.1021 means that *on average, each additional room is associated with an increase of $\$9,100$ in house price net of the other variables*. The confidence interval gives us a range of plausible values for this average change, about ($\$8,279, \$9,925$), definitely not chump change. 
# 
# In general, the $\hat{\beta_i}, i > 0$ can be interpreted as the following: "A one unit increase in $x_i$ is associated with, on average, a $\hat{\beta_i}$ increase/decrease in $y$ net of all other variables."
# 
# On the other hand, the interpretation for the intercept, $\hat{\beta}_0$ is the average of $y$ given that all of the independent variables $x_i$ are 0.

# ####  `statsmodels` formulas
# ***
# This formula notation will seem familiar to `R` users, but will take some getting used to for people coming from other languages or are new to statistics.
# 
# The formula gives instruction for a general structure for a regression call. For `statsmodels` (`ols` or `logit`) calls you need to have a Pandas dataframe with column names that you will add to your formula. In the below example you need a pandas data frame that includes the columns named (`Outcome`, `X1`,`X2`, ...), but you don't need to build a new dataframe for every regression. Use the same dataframe with all these things in it. The structure is very simple:
# 
# `Outcome ~ X1`
# 
# But of course we want to to be able to handle more complex models, for example multiple regression is doone like this:
# 
# `Outcome ~ X1 + X2 + X3`
# 
# In general, a formula for an OLS multiple linear regression is
# 
# `Y ~ X1 + X2 + ... + Xp`
# 
# This is the very basic structure but it should be enough to get you through the homework. Things can get much more complex. You can force statsmodels to treat variables as categorical with the `C()` function, call numpy functions to transform data such as `np.log` for extremely-skewed data, or fit a model without an intercept by including `- 1` in the formula. For a quick run-down of further uses see the `statsmodels` [help page](http://statsmodels.sourceforge.net/devel/example_formulas.html).
# 

# Let's see how our model actually fit our data. We can see below that there is a ceiling effect, we should probably look into that. Also, for large values of $Y$ we get underpredictions, most predictions are below the 45-degree gridlines. 

# <div class="span5 alert alert-info">
# <h3>Part 3 Checkup Exercise Set I</h3>
# 
# <p><b>Exercise:</b> Create a scatterplot between the predicted prices, available in `m.fittedvalues` (where `m` is the fitted model) and the original prices. How does the plot look? Do you notice anything interesting or weird in the plot? Comment on what you see.</p>
# </div>

# In[34]:


# your turn
sns.scatterplot(x=bos.PRICE, y=m.fittedvalues)
plt.title("Original Prices vs Predicted Prices")
plt.xlabel("Original Prices")
plt.ylabel("Predicted Prices")
plt.show()
# There is a strong relationship between actual and predicted prices except for price range 50 where differences seem slightly higher.


# ### Fitting Linear Regression using `sklearn`
# 

# In[35]:


from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE', axis = 1)

# This creates a LinearRegression object
lm = LinearRegression()
lm


# #### What can you do with a LinearRegression object? 
# ***
# Check out the scikit-learn [docs here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). We have listed the main functions here. Most machine learning models in scikit-learn follow this same API of fitting a model with `fit`, making predictions with `predict` and the appropriate scoring function `score` for each model.

# Main functions | Description
# --- | --- 
# `lm.fit()` | Fit a linear model
# `lm.predit()` | Predict Y using the linear model with estimated coefficients
# `lm.score()` | Returns the coefficient of determination (R^2). *A measure of how well observed outcomes are replicated by the model, as the proportion of total variation of outcomes explained by the model*

# #### What output can you get?

# In[ ]:


# Look inside lm object
# lm.<tab>


# Output | Description
# --- | --- 
# `lm.coef_` | Estimated coefficients
# `lm.intercept_` | Estimated intercept 

# ### Fit a linear model
# ***
# 
# The `lm.fit()` function estimates the coefficients the linear regression using least squares. 

# In[36]:


# Use all 13 predictors to fit linear regression model
lm.fit(X, bos.PRICE)


# <div class="span5 alert alert-info">
# <h3>Part 3 Checkup Exercise Set II</h3>
# 
# <p><b>Exercise:</b> How would you change the model to not fit an intercept term? Would you recommend not having an intercept? Why or why not? For more information on why to include or exclude an intercept, look [here](https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-what-is-regression-through-the-origin/).</p>
# 
# <p><b>Exercise:</b> One of the assumptions of the linear model is that the residuals must be i.i.d. (independently and identically distributed). To satisfy this, is it enough that the residuals are normally distributed? Explain your answer.</p>
# 
# <p><b>Exercise:</b> True or false. To use linear regression, $Y$ must be normally distributed. Explain your answer.</p>
# </div>
# 

# # your turn
# you can change the fit_inteercept to False for assuming not to fit an intercept term. That is not advisable however as in doing so you are saying the 
# y intercept is zero and the model is heavily biased towards the coefficient value and the purpose of reducing the mean squared error would not be satisfied.
# 
# Normal distribution of residuals wouldnt prove that they are independent and identically distributed. we could plot the residuals and see if they have a uniform distribution. Residuals are also the difference between the actual and predicted variable , and their independence lies on how independent the actual variables are.
# 
# Dependent value does not have to be normally distributed. However the residuals are needed to be.
# 
# 
# 

# ### Estimated intercept and coefficients
# 
# Let's look at the estimated coefficients from the linear model using `1m.intercept_` and `lm.coef_`.  
# 
# After we have fit our linear regression model using the least squares method, we want to see what are the estimates of our coefficients $\beta_0$, $\beta_1$, ..., $\beta_{13}$: 
# 
# $$ \hat{\beta}_0, \hat{\beta}_1, \ldots, \hat{\beta}_{13} $$
# 
# 

# In[37]:


print('Estimated intercept coefficient: {}'.format(lm.intercept_))


# In[38]:


print('Number of coefficients: {}'.format(len(lm.coef_)))


# In[39]:


# The coefficients
pd.DataFrame({'features': X.columns, 'estimatedCoefficients': lm.coef_})[['features', 'estimatedCoefficients']]


# ### Predict Prices 
# 
# We can calculate the predicted prices ($\hat{Y}_i$) using `lm.predict`. 
# 
# $$ \hat{Y}_i = \hat{\beta}_0 + \hat{\beta}_1 X_1 + \ldots \hat{\beta}_{13} X_{13} $$

# In[40]:


# first five predicted prices
lm.predict(X)[0:5]


# <div class="span5 alert alert-info">
# <h3>Part 3 Checkup Exercise Set III</h3>
# 
# <p><b>Exercise:</b> Histogram: Plot a histogram of all the predicted prices. Write a story about what you see. Describe the shape, center and spread of the distribution. Are there any outliers? What might be the reason for them? Should we do anything special with them?</p>
# 
# <p><b>Exercise:</b> Scatterplot: Let's plot the true prices compared to the predicted prices to see they disagree (we did this with `statsmodels` before).</p>
# 
# <p><b>Exercise:</b> We have looked at fitting a linear model in both `statsmodels` and `scikit-learn`. What are the advantages and disadvantages of each based on your exploration? Based on the information provided by both packages, what advantage does `statsmodels` provide?</p>
# </div>

# In[41]:


# your turn
plt.hist(lm.predict(X), bins=15)
plt.xlabel("House Price (in $1,000s)")
plt.ylabel("Count")
plt.xticks([x for x in range(0,50,5)])
plt.title("Predicted Housing Price Histogram")
print("Center: $%f      Spread (standard deviation): $%f" % (np.mean(lm.predict(X))*1000, np.std(lm.predict(X))*1000))

# They are normally distributed with the center and spread values below. There seems to be prediction values for hosuing prices less than 0
#that doesnt make sense.


# ### Evaluating the Model: Sum-of-Squares
# 
# The partitioning of the sum-of-squares shows the variance in the predictions explained by the model and the variance that is attributed to error.
# 
# $$TSS = ESS + RSS$$
# 
# #### Residual Sum-of-Squares (aka $RSS$)
# 
# The residual sum-of-squares is one of the basic ways of quantifying how much error exists in the fitted model. We will revisit this in a bit.
# 
# $$ RSS = \sum_{i=1}^N r_i^2 = \sum_{i=1}^N \left(y_i - \left(\beta_0 + \beta_1 x_i\right)\right)^2 $$

# In[42]:


print(np.sum((bos.PRICE - lm.predict(X)) ** 2))


# #### Explained Sum-of-Squares (aka $ESS$)
# 
# The explained sum-of-squares measures the variance explained by the regression model.
# 
# $$ESS = \sum_{i=1}^N \left( \hat{y}_i - \bar{y} \right)^2 = \sum_{i=1}^N \left( \left( \hat{\beta}_0 + \hat{\beta}_1 x_i \right) - \bar{y} \right)^2$$

# In[43]:


print(np.sum((lm.predict(X) - np.mean(bos.PRICE))**2))


# ### Evaluating the Model: The Coefficient of Determination ($R^2$)
# 
# The coefficient of determination, $R^2$, tells us the percentage of the variance in the response variable $Y$ that can be explained by the linear regression model.
# 
# $$ R^2 = \frac{ESS}{TSS} $$
# 
# The $R^2$ value is one of the most common metrics that people use in describing the quality of a model, but it is important to note that *$R^2$ increases artificially as a side-effect of increasing the number of independent variables.* While $R^2$ is reported in almost all statistical packages, another metric called the *adjusted $R^2$* is also provided as it takes into account the number of variables in the model, and can sometimes even be used for non-linear regression models!
# 
# $$R_{adj}^2 = 1 - \left( 1 - R^2 \right) \frac{N - 1}{N - K - 1} = R^2 - \left( 1 - R^2 \right) \frac{K}{N - K - 1} = 1 - \frac{\frac{RSS}{DF_R}}{\frac{TSS}{DF_T}}$$
# 
# where $N$ is the number of observations, $K$ is the number of variables, $DF_R = N - K - 1$ is the degrees of freedom associated with the residual error and $DF_T = N - 1$ is the degrees of the freedom of the total error.

# ### Evaluating the Model: Mean Squared Error and the $F$-Statistic
# ***
# The mean squared errors are just the *averages* of the sum-of-squares errors over their respective degrees of freedom.
# 
# $$MSR = \frac{ESS}{K}$$
# 
# $$MSE = \frac{RSS}{N-K-1}$$
# 
# **Remember:** Notation may vary across resources particularly the use of *R* and *E* in *RSS/ESS* and *MSR/MSE*. In some resources, E = explained and R = residual. In other resources, E = error and R = regression (explained). **This is a very important distinction that requires looking at the formula to determine which naming scheme is being used.**
# 
# Given the MSR and MSE, we can now determine whether or not the entire model we just fit is even statistically significant. We use an $F$-test for this. The null hypothesis is that all of the $\beta$ coefficients are zero, that is, none of them have any effect on $Y$. The alternative is that *at least one* $\beta$ coefficient is nonzero, but it doesn't tell us which one in a multiple regression:
# 
# $$H_0: \beta_i = 0, \mbox{for all $i$} \\
# H_A: \beta_i > 0, \mbox{for some $i$}$$ 
# 
# $$F = \frac{MSR}{MSE} = \left( \frac{R^2}{1 - R^2} \right) \left( \frac{N - K - 1}{K} \right)$$
#  
# Once we compute the $F$-statistic, we can use the $F$-distribution with $N-K$ and $K-1$ degrees of degrees of freedom to get a p-value.
# 
# **Warning!** The $F$-statistic mentioned in this section is NOT the same as the F1-measure or F1-value discused in Unit 7.

# <div class="span5 alert alert-info">
# <h3>Part 3 Checkup Exercise Set IV</h3>
# 
# <p>Let's look at the relationship between `PTRATIO` and housing price.</p>
# 
# <p><b>Exercise:</b> Try fitting a linear regression model using only the 'PTRATIO' (pupil-teacher ratio by town) and interpret the intercept and the coefficients.</p>
# 
# <p><b>Exercise:</b> Calculate (or extract) the $R^2$ value. What does it tell you?</p>
# 
# <p><b>Exercise:</b> Compute the $F$-statistic. What does it tell you?</p>
# 
# <p><b>Exercise:</b> Take a close look at the $F$-statistic and the $t$-statistic for the regression coefficient. What relationship do you notice? Note that this relationship only applies in *simple* linear regression models.</p>
# </div>

# In[44]:


# your turn
n = ols('PTRATIO ~ RM',bos).fit()
print(n.summary())


# In[61]:


lm.fit(bos.PTRATIO.values.reshape(-1,1) , bos.PRICE)
print("Coef: ", lm.coef_)
print("Intercept: ", lm.intercept_)
rsq=(lm.score(bos.PTRATIO.values.reshape(-1, 1), bos.PRICE))
print("R square: {}".format(rsq))
radj= 1-(1-rsq)*((len(bos.PTRATIO)-1)/(len(bos.PTRATIO)-1-1))
print ("R adj: ",radj)
fstat= (rsq/(1-rsq))*((len(bos.PTRATIO)-1-1)/1)
print("F statistic: ",fstat)
p = 1-stats.f.cdf(fstat, len(bos.PTRATIO), len(bos.PTRATIO))
print("P value: ",p)


# Only 25% of PTRATIO values can accurately predict the prices with a very high value of F statistic and P value less than 0.05 which does show statistical significance 

# <div class="span5 alert alert-info">
# <h3>Part 3 Checkup Exercise Set V</h3>
# 
# <p>Fit a linear regression model using three independent variables</p>
# 
# <ol>
# <li> 'CRIM' (per capita crime rate by town)
# <li> 'RM' (average number of rooms per dwelling)
# <li> 'PTRATIO' (pupil-teacher ratio by town)
# </ol>
# 
# <p><b>Exercise:</b> Compute or extract the $F$-statistic. What does it tell you about the model?</p>
# 
# <p><b>Exercise:</b> Compute or extract the $R^2$ statistic. What does it tell you about the model?</p>
# 
# <p><b>Exercise:</b> Which variables in the model are significant in predicting house price? Write a story that interprets the coefficients.</p>
# </div>

# In[62]:


# your turn
lm.fit(bos[["CRIM","RM","PTRATIO"]], bos.PRICE)
rsq=lm.score(bos[["CRIM","RM","PTRATIO"]], bos.PRICE)
print("Coefficient Print\nCRIM: %f   RM: %f   PTRATIO: %f" % (lm.coef_[0], lm.coef_[1], lm.coef_[2]))
print("R square: {}".format(rsq))
radj= 1-(1-rsq)*((len(bos.PTRATIO)-1)/(len(bos.PTRATIO)-3-1))
print ("R adj: ",radj)
fstat= (rsq/(1-rsq))*((len(bos.PTRATIO)-3-1)/3)
print("F statistic: ",fstat)
p = 1-stats.f.cdf(fstat, len(bos.PTRATIO), len(bos.PTRATIO))
print("P value: ",p)


# 59 percent of the predictions can be explained by +7.38 times PTRATIO, -1.06 times RM and -0.20 times CRIM with a p value less than 0.05 making it statistically significant.

# ## Part 4: Comparing Models

# During modeling, there will be times when we want to compare models to see which one is more predictive or fits the data better. There are many ways to compare models, but we will focus on two.

# ### The $F$-Statistic Revisited
# 
# The $F$-statistic can also be used to compare two *nested* models, that is, two models trained on the same dataset where one of the models contains a *subset* of the variables of the other model. The *full* model contains $K$ variables and the *reduced* model contains a subset of these $K$ variables. This allows us to add additional variables to a base model and then test if adding the variables helped the model fit.
# 
# $$F = \frac{\left( \frac{RSS_{reduced} - RSS_{full}}{DF_{reduced} - DF_{full}} \right)}{\left( \frac{RSS_{full}}{DF_{full}} \right)}$$
# 
# where $DF_x = N - K_x - 1$ where $K_x$ is the number of variables in model $x$.

# ### Akaike Information Criterion (AIC)
# 
# Another statistic for comparing two models is AIC, which is based on the likelihood function and takes into account the number of variables in the model.
# 
# $$AIC = 2 K - 2 \log_e{L}$$
# 
# where $L$ is the likelihood of the model. AIC is meaningless in the absolute sense, and is only meaningful when compared to AIC values from other models. Lower values of AIC indicate better fitting models.
# 
# `statsmodels` provides the AIC in its output.

# <div class="span5 alert alert-info">
# <h3>Part 4 Checkup Exercises</h3>
# 
# <p><b>Exercise:</b> Find another variable (or two) to add to the model we built in Part 3. Compute the $F$-test comparing the two models as well as the AIC. Which model is better?</p>
# </div>

# In[63]:


m1 = ols('PRICE ~ CRIM + RM + PTRATIO',bos).fit()
print(m1.summary())
m2 = ols('PRICE ~ CRIM + RM + PTRATIO + AGE + DIS',bos).fit()
print(m2.summary())


# R square or adjusted R square value seems to be higher for model 2, however this could also be due to the increase in independent variables and not accuracy. Model 2 has a lesser AIC value suggesting higher performance than model 1 with both having statistical significance, with Model 1 having higher F statistic value means higher variance.

# 
# ## Part 5: Evaluating the Model via Model Assumptions and Other Issues
# ***
# Linear regression makes several assumptions. It is always best to check that these assumptions are valid after fitting a linear regression model.
# 
# <div class="span5 alert alert-danger">
# <ul>
#   <li>**Linearity**. The dependent variable $Y$ is a linear combination of the regression coefficients and the independent variables $X$. This can be verified with a scatterplot of each $X$ vs. $Y$ and plotting correlations among $X$. Nonlinearity can sometimes be resolved by [transforming](https://onlinecourses.science.psu.edu/stat501/node/318) one or more independent variables, the dependent variable, or both. In other cases, a [generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model) or a [nonlinear model](https://en.wikipedia.org/wiki/Nonlinear_regression) may be warranted.</li>
#   <li>**Constant standard deviation**. The SD of the dependent variable $Y$ should be constant for different values of X. We can check this by plotting each $X$ against $Y$ and verifying that there is no "funnel" shape showing data points fanning out as $X$ increases or decreases. Some techniques for dealing with non-constant variance include weighted least squares (WLS), [robust standard errors](https://en.wikipedia.org/wiki/Heteroscedasticity-consistent_standard_errors), or variance stabilizing transformations.
#     </li>
#   <li> **Normal distribution for errors**.  The $\epsilon$ term we discussed at the beginning are assumed to be normally distributed. This can be verified with a fitted values vs. residuals plot and verifying that there is no pattern, and with a quantile plot.
#   $$ \epsilon_i \sim N(0, \sigma^2)$$
# Sometimes the distributions of responses $Y$ may not be normally distributed at any given value of $X$.  e.g. skewed positively or negatively. </li>
# <li> **Independent errors**.  The observations are assumed to be obtained independently.
#     <ul>
#         <li>e.g. Observations across time may be correlated
#     </ul>
# </li>
# </ul>  
# 
# </div>
# 
# There are some other issues that are important investigate with linear regression models.
# 
# <div class="span5 alert alert-danger">
# <ul>
#   <li>**Correlated Predictors:** Care should be taken to make sure that the independent variables in a regression model are not too highly correlated. Correlated predictors typically do not majorly affect prediction, but do inflate standard errors of coefficients making interpretation unreliable. Common solutions are dropping the least important variables involved in the correlations, using regularlization, or, when many predictors are highly correlated, considering a dimension reduction technique such as principal component analysis (PCA).
#   <li>**Influential Points:** Data points that have undue influence on the regression model. These points can be high leverage points or outliers. Such points are typically removed and the regression model rerun.
# </ul>
# </div>
# 

# <div class="span5 alert alert-info">
# <h3>Part 5 Checkup Exercises</h3>
# 
# <p>Take the reduced model from Part 3 to answer the following exercises. Take a look at [this blog post](http://mpastell.com/2013/04/19/python_regression/) for more information on using statsmodels to construct these plots.</p>
#     
# <p><b>Exercise:</b> Construct a fitted values versus residuals plot. What does the plot tell you? Are there any violations of the model assumptions?</p>
# 
# <p><b>Exercise:</b> Construct a quantile plot of the residuals. What does the plot tell you?</p>
# 
# <p><b>Exercise:</b> What are some advantages and disadvantages of the fitted vs. residual and quantile plot compared to each other?</p>
# 
# <p><b>Exercise:</b> Identify any outliers (if any) in your model and write a story describing what these outliers might represent.</p>
# 
# <p><b>Exercise:</b> Construct a leverage plot and identify high leverage points in the model. Write a story explaining possible reasons for the high leverage points.</p>
# 
# <p><b>Exercise:</b> Remove the outliers and high leverage points from your model and run the regression again. How do the results change?</p>
# </div>

# In[67]:


# Your turn.
plt.scatter(lm.predict(bos[["CRIM","RM","PTRATIO"]]), bos.PRICE - lm.predict(bos[["CRIM","RM","PTRATIO"]]))
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()

plt.hist(bos.PRICE - lm.predict(bos[["CRIM","RM","PTRATIO"]]),bins = 30)
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Residual Histogram")
plt.show()


# There is clearly a relationship between residuals and the predicted values , When the residuals center on zero they indicate that the model’s predictions are correct on average rather than systematically too high or low. Regression also assumes that the residuals follow a normal distribution and that the degree of scattering is the same for all fitted values. There seems to be outliers for Price greater than 20

# In[68]:


stats.probplot(bos.PRICE - lm.predict(bos[["CRIM","RM","PTRATIO"]]), fit=True, plot=plt)
plt.show()
# residuals showing normality in most cases except for prices above 20 as noted earlier


# In[69]:


import statsmodels
plt.rcParams["figure.figsize"] = (20,20)
statsmodels.graphics.regressionplots.influence_plot(m1)
plt.show()


# We have high leverage points in the extreme right of the x axis which means those rows which differ significantly from other independent rows
# and ones with high residual values after 4.Lets evaluate the performance by removing them.

# In[70]:


bos1 =bos.drop(bos.index[[414,410,405,418,380,367,370,369,365,371,372,368]]).reindex()


# In[72]:


# your turn
lm.fit(bos1[["CRIM","RM","PTRATIO"]], bos1.PRICE)
rsq=lm.score(bos1[["CRIM","RM","PTRATIO"]], bos1.PRICE)
print("Coefficient Print\nCRIM: %f   RM: %f   PTRATIO: %f" % (lm.coef_[0], lm.coef_[1], lm.coef_[2]))
print("R square: {}".format(rsq))
radj= 1-(1-rsq)*((len(bos1.PTRATIO)-1)/(len(bos1.PTRATIO)-3-1))
print ("R adj: ",radj)
fstat= (rsq/(1-rsq))*((len(bos1.PTRATIO)-3-1)/3)
print("F statistic: ",fstat)
p = 1-stats.f.cdf(fstat, len(bos1.PTRATIO), len(bos1.PTRATIO))
print("P value: ",p)
plt.scatter(bos1.PRICE, lm.predict(bos1[["CRIM","RM","PTRATIO"]]))
plt.xlabel("Observed Price")
plt.ylabel("Predicted Price")
plt.title("Observed Price vs Predicted Price")


# In[73]:


m1 = ols('PRICE ~ CRIM + RM + PTRATIO',bos).fit()
print(m1.summary())
m3 = ols('PRICE ~ CRIM + RM + PTRATIO',bos1).fit()
print(m3.summary())


# R square /adjusted r square value has increased from 0.59 to .72 after outlier removal suggesting further increase in dependent and independent variable relationships. Values seems to be statistically significant as well with a p value < 0.05 and lesser AIC value.
# 
# Although it does not mean outlier removed model best represents the data, its just the best fit linear model.
