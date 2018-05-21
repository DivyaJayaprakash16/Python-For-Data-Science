
# coding: utf-8

# # Bitcoin Market Price Prediction
# ## Divya Jayaprakash

# Start by importing all the libraries and the training and test datasets. 

# In[1]:


get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, validation_curve, StratifiedKFold, GridSearchCV

data = pd.read_csv('bitcoin_dataset.csv')
test = pd.read_csv('test_set.csv')


# In[2]:


print(data.columns)
print(test.columns)


# ## Data Cleaning and Imputation

# The next cell renames all the columns other than the date column

# In[3]:


for col in data.columns[1:]:
    data.rename(columns={col:col[4:]}, inplace = True)

for col in test.columns[1:]:
    test.rename(columns={col:col[4:]}, inplace=True)


# In[4]:


print(data.columns)
print(test.columns)


# Set the index to date. This can be used to plot according to time and use for imputation.

# In[5]:


data.head()
data["Date"] = pd.to_datetime(data["Date"])
test["Date"] = pd.to_datetime(test["Date"])
data = data.set_index("Date")
test = test.set_index("Date")


# Check the percentage of missing values in the training data and the test data

# In[6]:


print(100*data.isnull().sum().sum() / data.shape[0]) #4% overall nulls in train
print(100*test.isnull().sum().sum() / test.shape[0]) #0% here


# Print the percentage of missing values in each column

# In[7]:


print(100*data.isnull().sum() / data.shape[0]) 


# Plot the missing values columns to see the trend(or pattern) to decide on how to impute

# In[8]:


sns.set_style("white")
data[["total_bitcoins","trade_volume","blocks_size",
      "median_confirmation_time", "difficulty","transaction_fees"]].plot(subplots=True, figsize=(15, 15)); plt.legend(loc='best')


# The ColsToImpute below has features with mostly a linear trend near the missing values. So, we are using a linear interpolation method with time to impute them. <br>
# After this, only the median_confirmation_time column will have missing values and it is reasonable to fill them using the mean value because it has most of it's values 0 and 47 with mean around 7.5

# In[9]:


ColsToImpute = ["total_bitcoins","trade_volume","blocks_size","difficulty","transaction_fees"]
data[ColsToImpute] = data[ColsToImpute].interpolate(method="time")
data = data.fillna(data.mean())


# Now, we don't require the date column so reset the index and remove the date column

# In[10]:


data = data.reset_index()
test = test.reset_index()
data = data.drop('Date', 1)
test = test.drop('Date', 1)
print(data.columns)
print(test.columns)


# Split the "data" dataframe into dependent and independent variables

# In[11]:


X_btc = data[data.columns[1:len(data.columns)]]
y_btc = data['market_price']


# Split into training and test dataset

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X_btc, y_btc, test_size=0.3, random_state=1)


# We are using scaled data to run most of our models for better computation and convergence

# In[13]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Model Building

# ### KNN Regressor

# Run the KNN Regressor for K from 1 to 19;
# For K = 2 we are getting the best test accuracy. So the train accuracy is 0.9979 and test accuracy is 0.9944 for KNN regressor.

# In[14]:


for num_neighbors in range(1, 20):
    knnreg = KNeighborsRegressor(n_neighbors = num_neighbors).fit(X_train_scaled, y_train)

    #print(knnreg.predict(X_test_scaled))
    print('R-squared train score for {} neighbors: {:.4f}'
         .format(num_neighbors, knnreg.score(X_train_scaled, y_train)))
    print('R-squared test score for {} neighbors: {:.4f}'
         .format(num_neighbors, knnreg.score(X_test_scaled, y_test)))


# ### Linear Regression

# Next, we used linear regression to predict the market price. The train accuracy is 0.999964 and test accuracy is 0.999936. As we can see that Linear Regression performs better we will not consider KNN for further analysis.

# In[15]:


linregscaled = LinearRegression().fit(X_train_scaled, y_train)

print('R-squared score (training): {:.6f}'
     .format(linregscaled.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.6f}'
     .format(linregscaled.score(X_test_scaled, y_test)))


# ### Ridge Regression

# Next, we try Ridge regression with different values of alpha. This seems to perform good with very low values of alpha.

# In[16]:


print('Ridge regression: Trying different values of alpha')
for this_alpha in [0, 0.001, 0.01, 1, 10, 20, 50, 100, 1000]:
    linRidge = Ridge(alpha=this_alpha).fit(X_train_scaled, y_train)
    r2_train = linRidge.score(X_train_scaled, y_train)
    r2_test = linRidge.score(X_test_scaled, y_test)
    num_coeff_big = np.sum(abs(linRidge.coef_) > 1.0)
    print('Alpha = {:.7f}\nnum abs(coeff) > 1.0: {}, r-squared training: {:.7f}, r-squared test: {:.7f}\n'
         .format(this_alpha, num_coeff_big, r2_train, r2_test))


# ### Lasso Regression

# Next, we try Lasso with alpha of 0.01 to start with and we can see that this does a pretty good job and gives similar result to Linear Regression.

# In[17]:


linlasso = Lasso(alpha=0.01, max_iter=1000).fit(X_train_scaled, y_train)

print('R-squared score (training): {:.7f}'
     .format(linlasso.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.7f}\n'
     .format(linlasso.score(X_test_scaled, y_test)))
print('Features with non-zero weight (sorted by absolute magnitude):')

for e in sorted (list(zip(list(X_train), linlasso.coef_)),
                key = lambda e: -abs(e[1])):
    if e[1] != 0:
        print('\t{}, {:.3f}'.format(e[0], e[1]))


# Next, we try Lasso with different values of alpha to see the behaviour by varying alpha and when alpha is 0.01 it is performing the best on the test set.

# In[18]:


print('Trying Lasso for different values of alpha')

for alpha in [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 3, 5, 10]:
    linLasso = Lasso(alpha, max_iter=10000).fit(X_train_scaled, y_train)
    r2_train = linLasso.score(X_train_scaled, y_train)
    r2_test = linLasso.score(X_test_scaled, y_test)
    
    print('Alpha = {:.4f}\nFeatures kept: {}\nr-squared training: {:.7f}, r-squared test: {:.7f}\n'
         .format(alpha, np.sum(linLasso.coef_ != 0), r2_train, r2_test))


# ### Polynomial Regression

# Next, let us try polynomial regression because a lot of variables seamt to have a non-linear trend and kind of parabolic trend. As we can see in the result below, this is performing better than even linear regression with a better accuracy on both train and test set.

# In[19]:


poly = PolynomialFeatures(degree=2)
X_F1_poly = poly.fit_transform(X_btc)

scaler = MinMaxScaler()
X_poly_scaled = scaler.fit_transform(X_F1_poly)

X_train, X_test, y_train, y_test = train_test_split(X_poly_scaled, y_btc, test_size = 0.3, random_state = 1)

polyreg = LinearRegression().fit(X_train, y_train)

print('(poly deg 2) R-squared score (training): {:.7f}'
     .format(polyreg.score(X_train, y_train)))
print('(poly deg 2) R-squared score (test): {:.7f}\n'
     .format(polyreg.score(X_test, y_test)))


# ### Polynomial Regression with Ridge and Lasso

# Next, we run Ridge with polynomial with Ridge and after seeing the result of the cell below we can see that if does not perform that better on the test set. Let us try Lasso next

# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X_poly_scaled, y_btc, test_size = 0.3, random_state = 1)

polyRidgeReg = Ridge(alpha = 0.1).fit(X_train, y_train)

print('Polynomial Regression with Regularization')

print('(poly deg 2 + ridge) R-squared score (training): {:.7f}'
     .format(polyRidgeReg.score(X_train, y_train)))
print('(poly deg 2 + ridge) R-squared score (test): {:.7f}'
     .format(polyRidgeReg.score(X_test, y_test)))

polyLassoReg = Lasso(alpha = 0.001).fit(X_train, y_train)

print('(poly deg 2 + lasso) R-squared score (training): {:.7f}'
     .format(polyLassoReg.score(X_train, y_train)))
print('(poly deg 2 + lasso) R-squared score (test): {:.7f}'
     .format(polyLassoReg.score(X_test, y_test)))


# ## Evaluation

# ### Cross Validation

# There is imbalance in few columns so the validation set is not getting split well. So, the scores are either exploding or vanishing.

# In[21]:


scaler = MinMaxScaler()
X_btc_scaled = scaler.fit_transform(X_btc)

scores = cross_val_score(linregscaled, X_btc_scaled, y_btc, cv = 3)
print("Linear Regression Cross validation scores: {}".format(scores))
print("Average cross-validation scroe: {:.2f}".format(scores.mean()))


# Let us use StratifiedKFold to do a better sampling for the trainset and try cross validation for Linear, Ridge and Lasso Regression.

# In[22]:


import warnings; warnings.simplefilter('ignore')

scores = cross_val_score(linregscaled, X_btc_scaled, y_btc, cv = StratifiedKFold(10, shuffle = True, random_state = 10))
print("Cross validation scores: {}".format(scores))
print("Average CV scores for linear regression: {:.7f}".format(scores.mean()))

scores = cross_val_score(linlasso, X_btc_scaled, y_btc, cv = StratifiedKFold(10, shuffle = True, random_state = 10))
print("Cross validation scores: {}".format(scores))
print("Average CV scores for Lasso Regression: {:.7f}".format(scores.mean()))

scores = cross_val_score(linRidge, X_btc_scaled, y_btc, cv = StratifiedKFold(10, shuffle = True, random_state = 10))
print("Cross validation scores: {}".format(scores))
print("Average CV scores for Ridge Regression: {:.7f}".format(scores.mean()))


# The results from above are still not better than polynomial regression. So, let us run cross validation on different polynomial regression with plain, lasso and ridge. We can see normal polynomial regression and Lasso perform similar to each other.

# In[23]:


scores = cross_val_score(polyreg, X_poly_scaled, y_btc, cv = StratifiedKFold(10, shuffle = True, random_state = 10))
print("Cross validation scores: {}".format(scores))
print("Average CV scores for Polynomial Regression: {:.7f}".format(scores.mean()))

scores = cross_val_score(polyRidgeReg, X_poly_scaled, y_btc, cv = StratifiedKFold(10, shuffle = True, random_state = 10))
print("Cross validation scores: {}".format(scores))
print("Average CV scores for Polynomial Ridge Regression: {:.7f}".format(scores.mean()))

scores = cross_val_score(polyLassoReg, X_poly_scaled, y_btc, cv = StratifiedKFold(10, shuffle = True, random_state = 10))
print("Cross validation scores: {}".format(scores))
print("Average CV scores for Polynomial Lasso Regression: {:.7f}".format(scores.mean()))


# ### Grid Search

# As with polynomial we will use a lot of features and Lasso is giving the same result. It is better to use Lasso for the prediction and we will use grid search to find the best value of alpha.

# In[24]:


param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(polyLassoReg, param_grid, cv = StratifiedKFold(10, shuffle = True, random_state = 10))
grid_search.fit(X_poly_scaled, y_btc)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.7f}".format(grid_search.best_score_))


# Narrow down the search near 0.001 to find a better value of alpha.

# In[25]:


param_grid = {'alpha': [0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002]}

grid_search = GridSearchCV(polyLassoReg, param_grid, cv = StratifiedKFold(10, shuffle = True, random_state = 10))
grid_search.fit(X_poly_scaled, y_btc)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.7f}".format(grid_search.best_score_))


# SVM is taking a lot of time to run and is not giving good results. So, we have just written the code and not run. We will use the Polynomial Lasso regression with the alpha of 0.0012 to predict the values for the final test set.

# In[ ]:


#SVM

X_train, X_test, y_train, y_test = train_test_split(X_btc, y_btc, random_state = 1, test_size = 0.3)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR(kernel = 'linear', epsilon = 0.001, C = 0.01)
svr.fit(X_train_scaled, y_train)
print(svr.score(X_train_scaled, y_train))
print(svr.score(X_test_scaled, y_test))

svr = SVR(kernel = 'rbf', gamma = 'auto', epsilon = 0.001, C = 0.01)
svr.fit(X_train_scaled, y_train)
print(svr.score(X_train_scaled, y_train))
print(svr.score(X_test_scaled, y_test))


# ## Final Test Predicted values

# In[26]:


poly = PolynomialFeatures(degree=2)
X_F1_poly = poly.fit_transform(X_btc)
X_test_poly = poly.fit_transform(test)

scaler = MinMaxScaler()
X_poly_scaled = scaler.fit_transform(X_F1_poly)

polyReg = Lasso(alpha = 0.0012).fit(X_poly_scaled, y_btc)

Final_test_scaled = scaler.transform(X_test_poly)
polyReg.predict(Final_test_scaled)


# We are also putting results for Linear Regression so we can also consider those values for better decision making.

# In[28]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_btc)

test_scale = scaler.transform(test)
linregscaled.predict(test_scale)


# In[31]:


test.shape


# In[32]:


X_btc.shape


# In[33]:


X_train.shape


# In[35]:


X_train


# In[36]:


test

