# -*- coding: utf-8 -*-
"""

"""

# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin


# import dataset, set target variable
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
target = train["SalePrice"]

# drop target variable from training set
train = train.drop("SalePrice", 1)

# create additional variable to discern training and test 
train['training'] = True
test['training'] = False
train.head(3)
test.head(3)

# combine both datasets
full = pd.concat([train, test])

# analysis of data contents and shape
print("Column names of data: {}".format(full.columns))
print("Data types of colums: {}".format(full.columns.to_series().groupby(full.dtypes).groups))
print("Shape of full data: {}".format(full.shape))
print("Shape of train data: {}".format(train.shape))
print("Shape of test data: {}".format(test.shape))


# check for null data in 'SalePrice'
target.isnull().values.sum()

# examine skewness of target
import seaborn as sns
sns.distplot(target, hist = True, kde = False, color = 'green',
             bins = 50, hist_kws = {'edgecolor':'black'})
plt.title('Price Distribution')
plt.ylabel('Frequency')

# transform price based on skewness
target = np.log(target)
target.head(3)

# check for null values in the rest of the data
print("Columns with null values: {}".format(full.columns[full.isnull().any()]))
print("Number of null values by variable: {}".format(full.isnull().sum()))

# use TransformerMixin to impute
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """
        --> Missing categorical values are imputed with most frequent value
        --> Missing numerical values are imputed with median of variable
        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
X = pd.DataFrame(full)
complete_full = DataFrameImputer().fit_transform(X)
print(complete_full.head(3))

# check for missing values after imputation
print("Columns with null values after imputation: {}".format(complete_full.columns[complete_full.isnull().any()]))
print("Number of null values by variable: {}".format(complete_full.isnull().sum()))

# dummy variable code
complete_full = pd.get_dummies(complete_full)

# split joined dataset back into train and validation data
train = complete_full[complete_full['training'] == True]
test = complete_full[complete_full['training'] == False]

# drop training variable from train and validation datasets
train = train.drop('training', 1)
test = test.drop('training', 1)
print(train.shape,test.shape)
train.head(3)
test.head(3)


# split training set into training and test - 75% / 25%
X_train, X_test, y_train, y_test = train_test_split(train, target, random_state = 0)
print("Shape of training treatment data: {}".format(X_train.shape))
print("Shape of training response data: {}".format(y_train.shape))
print("Shape of testing treatment data: {}".format(X_test.shape))
print("Shape of testing response data: {}".format(y_test.shape))

# RANDOM FOREST TOP TEN FEATURES DATASET
rfm_train = X_train[['LotArea','OverallQual','OverallCond','YearBuilt','BsmtFinSF1','TotalBsmtSF','1stFlrSF','GrLivArea','GarageCars','GarageArea']]
rfm_test = X_test[['LotArea','OverallQual','OverallCond','YearBuilt','BsmtFinSF1','TotalBsmtSF','1stFlrSF','GrLivArea','GarageCars','GarageArea']]
# SelectKBest - import
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# select best k features using f value as the metric
selector = SelectKBest(f_regression, k = 50)
# fit to model
selector_fit = selector.fit_transform(X_train, y_train)

# combine selected features and f-values into dataframe and print in descending order
features = X_train.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
features_scores = list(zip(features, scores))
k_best_df = pd.DataFrame(data = features_scores, columns = ['Features', 'F-Value'])
k_best_df = k_best_df.sort_values(['F-Value', 'Features'], ascending = [False, True])
print(k_best_df)

# RECURSIVE FEATURE ELIMINATION


# use RandomForestClassifier to find top ten features relating to housing prices
# Used 11 features due to ID being in dataset
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor


#rfe = RFE(RandomForestRegressor(n_estimators=50, random_state=42), n_features_to_select=11)

#rfe.fit(X_train, y_train)

 #create dataframe of feature and ranking. Top 11 have '1' in rfe.ranking_ array
#rfe_features_rank = pd.DataFrame({'feature':X_train.columns, 'score':rfe.ranking_})
#compose list of highest ranked features
#top_ten_features = rfe_features_rank[rfe_features_rank['score'] == 1]['feature'].values
#print(top_ten_features)


# RANDOM FOREST REGRESSOR

from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

rnd_forest_model= RandomForestRegressor()

rnd_forest_model.fit(X_train, y_train)
y_predict = rnd_forest_model.predict(X_test)
y_prediction = pd.DataFrame(y_predict, columns=["SalePrice"])
print(y_prediction.head(10))
plt.scatter(y_test, y_predict)

print("Accuracy on test set: {:.3f}".format(rnd_forest_model.score(X_test, y_test)))

print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
print('MSE: ', mean_absolute_error(y_test, y_predict))


# RANDOM FOREST USING TOP TEN FEATURES

rnd_forest_model.fit(rfm_train, y_train)
rnd_top_y_predict =rnd_forest_model.predict(rfm_test)
rnd_top_y_prediction = pd.DataFrame(rnd_top_y_predict, columns=["SalePrice"])
print(rnd_top_y_prediction.head(10))
plt.scatter(y_test, rnd_top_y_predict)

print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, rnd_top_y_predict)))
print('MSE: ', mean_absolute_error(y_test, rnd_top_y_predict))


# PERFORM CROSS VALIDATION TO CHECK FOR OPTIMUM PARAMETERS

parameters={"n_estimators":[5,10,25,50, 70, 90, 125, 150],
             "max_features":[50,70,100,120, 150, 200]}

grid=GridSearchCV(rnd_forest_model, parameters)

grid.fit(X_train, y_train)
grid.score(X_test, y_test)

print(grid.best_params_)

# predictions = grid.predict(X_test)
# predictions = pd.DataFrame(predictions, columns=["SalePrice"])
# print(predictions.head(10))

# XGBoost Model
# import necessary features
import xgboost as xgb
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV
xgb_regressor = xgb.XGBRegressor(random_state = 21)

# use c-v to find optimum hyperparameters
# shuffle our data for cross validation
# commented out due to time it takes to run
# cv_sets_xgb = ShuffleSplit(random_state = 0)
# set parameter options
# parameters_xgb = {'n_estimators':[7000, 7500, 8000],
                  # 'learning_rate':[0.05, 0.06, 0.07],
                  # 'max_depth':[3, 5, 7],
                  # 'min_child_weight':[1.0, 1.5, 2.0]}
# scorer_xgb = make_scorer(r2_score)
# grid_obj_xgb = RandomizedSearchCV(xgb_regressor, parameters_xgb,
                                  # scoring = scorer_xgb, cv = cv_sets_xgb,
                                  # random_state = 1)
# grid_fit_xgb = grid_obj_xgb.fit(X_train, y_train)
# xgb_opt = grid_fit_xgb.best_estimator_
# get best parameters
# grid_fit_xgb.best_params_

# XGBoost with optimal tuned parameters from above code chunk
xgb_opt = xgb.XGBRegressor(
        learning_rate = 0.06, max_depth = 3, 
        min_child_rate = 1.0, n_estimators = 7000,
        seed = 21, silent = 1)

# fit model and predict
xgb_opt.fit(X_train, y_train)
xgb_opt_predict = xgb_opt.predict(X_test)

# get MSE of model
from sklearn.metrics import mean_squared_error
xgb_mse = mean_squared_error(y_test, xgb_opt_predict)

print(xgb_mse)






