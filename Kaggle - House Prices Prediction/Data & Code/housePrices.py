import pandas as pd
import numpy as np
import scipy
#from sklearn import cross_validation
import sklearn.dummy
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
import functools
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR
from sklearn import ensemble

# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("train.csv")
    testDF = pd.read_csv("test.csv")

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    #doTraining(trainInput, trainOutput, predictors)
    doTesting(trainInput, testInput, trainOutput, testIDs, predictors)

'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doTesting(trainInput, testInput, trainOutput, testIDs, predictors):
    #alg = linear_model.LinearRegression()
    #alg = linear_model.Ridge(alpha = 0.5)
    alg = GradientBoostingRegressor()

    # Train the algorithm using all the training data
    alg.fit(trainInput[predictors], trainOutput)

    # Make predictions using the test set.
    predictions = alg.predict(testInput[predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('testResults.csv', index=False)

    #Preparing normalized datasets for performing PCA
    #trainDF.to_csv('mod_train.csv', index = False)    
    #testDF.to_csv('norm_test.csv', index = False)

'''
Runs the algorithm on the training set.
'''
def doTraining(trainInput, trainOutput, predictors):
    scoring = 'r2'
    #scoring = 'rmse'

    #print("DummyRegressor:", doCrossValidation(trainDF, predictors, sklearn.dummy.DummyRegressor(strategy='mean'))) # Scores about 0, as expected for r^2 scoring

    # For now, this first group is pretty much all tied for best:
    
    #print("LinearRegression:", doCrossValidation(trainDF, predictors, linear_model.LinearRegression()))
    #print("QuantileRegression:", doCrossValidation(trainDF, predictors,  GradientBoostingRegressor(loss='quantile', alpha= 0.5)))
    #print("Ridge:", doCrossValidation(trainInput, trainOutput, linear_model.Ridge(alpha = 0.5), scoring))

    #alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    #cv_ridge = [doCrossValidation(trainDF, predictors, linear_model.Ridge(alpha = alpha)).mean() for alpha in alphas]
    #print(cv_ridge)
    #print("Best alpha:", alphas[cv_ridge.index(min(cv_ridge))])

    #print("Lasso:", doCrossValidation(trainInput, trainOutput, linear_model.Lasso(alpha = 0.2, max_iter=50000), scoring))  # Won't converge!

    #print("RandomForestRegressor:", doCrossValidation(trainDF, predictors, ensemble.RandomForestRegressor()))

    #print("BaggingRegressor, LinearRegression:", doCrossValidation(trainDF, predictors, ensemble.BaggingRegressor(linear_model.LinearRegression())))

    print("GradientBoostingRegressor:", doCrossValidation(trainInput[predictors], trainOutput, ensemble.GradientBoostingRegressor(), scoring))  # Best so far

    #print("RandomForestRegressor:", doCrossValidation(trainDF, predictors, ensemble.RandomForestRegressor()))  # a little worse

    # print("LassoLars:", doCrossValidation(trainDF, predictors, linear_model.LassoLars(alpha = 0.1)))
    # print("OrthogonalMatchingPursuit:", doCrossValidation(trainDF, predictors, linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs = 7)))
    # print("BayesianRidge:", doCrossValidation(trainDF, predictors, linear_model.BayesianRidge()))

    # Don't know how to do these:
    # print("Perceptron:", doCrossValidation(trainDF, predictors, linear_model.?????()))

    # For now, the following are significantly worse:
    # print("ElasticNet:", doCrossValidation(trainDF, predictors, linear_model.ElasticNet(alpha = 0.1)))
    # print("ARDRegression:", doCrossValidation(trainDF, predictors, linear_model.ARDRegression(compute_score=True)))
    # print("SGDRegressor:", doCrossValidation(trainDF, predictors, linear_model.SGDRegressor()))
    # print("PassiveAggressiveRegressor:", doCrossValidation(trainDF, predictors, linear_model.PassiveAggressiveRegressor()))
    # print("DecisionTreeRegressor:", doCrossValidation(trainDF, predictors, DecisionTreeRegressor(max_depth=4)))

    # And these are terrible for now:
    #print("Support Vector Regression, linear:", doCrossValidation(trainDF, predictors, SVR(C=1.0, epsilon=0.2, kernel='linear')))
    # Predicts every price to be around 160k...  ?
    #print("Support Vector Regression, rbf:", doCrossValidation(trainDF, predictors, SVR(C=1.0, epsilon=0.2, kernel='rbf')))

# ============================================================================
# Data cleaning - conversion, normalization

def printTopN(df, n, attrs):
    for rowID in range(n):
        print("----------------- Row", rowID, "-------------------")
        for attr in attrs:
            print(attr, ":", df.iloc[rowID][attr])

def __handleMissingAttributes(allInputs):
    # print("Missing values:", getAttrsWithMissingValues(allInputs))
    #['MSZoning', 'LotFrontage', 'Alley', 'Utilities', 'Exterior1st',
    #   'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond',
    #   'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
    #   'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical', 'BsmtFullBath',
    #   'BsmtHalfBath', 'KitchenQual', 'Functional', 'FireplaceQu',
    #   'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
    #   'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',
    #   'SaleType'],

    # This option is appropriate for numerical attributes where NA can be interpreted as 0
    replaceWithZero = ['LotFrontage'      # There are no 0s in the data, so NA probably means 0 linear feet connected.  
                      ]
    
    # This option is appropriate for categorical attributes that must have some value - where "not applicable" doesn't make sense
    replaceWithMode = ['MSZoning',        # Every house must be zoned somehow, so if missing, then put most common 
                       'Exterior1st',     # Every house has one exterior covering, so if missing, then put most common
                       'Exterior2nd',     # Only one house, in test set, is missing this and Exterior1st, so just put most common
                       'MasVnrType',      # There's already a None option, so NA must really mean missing
                       'Electrical',      # The only NA is a house built in 2007. Surely it has electricity, so the value must just be missing
                       'KitchenQual',     # Just one house has NA - surely it has a kitchen, so replace with mode
                       'Functional',      # Just two in test set have NA. There must be some functional rating.
                       'SaleType'         # Just one in the test set has NA. There must be some sale type.
                      ]

    # This option is appropriate for numerical attributes that must have some value - where "not applicable" doesn't make sense
    replaceWithMean = ['MasVnrArea',      # There are already 0s, so NA must really mean missing 
                       'BsmtFinSF1',      # There are already 0s, so NA must really mean missing
                       'BsmtFinSF2',      # There are already 0s, so NA must really mean missing
                       'BsmtUnfSF',       # There are already 0s, so NA must really mean missing
                       'TotalBsmtSF',     # There are already 0s, so NA must really mean missing
                       'BsmtFullBath',    # There are already 0s, so NA must really mean missing
                       'BsmtHalfBath',    # There are already 0s, so NA must really mean missing
                       'GarageCars',      # There are already 0s, so NA must really mean missing
                       'GarageArea'       # There are already 0s, so NA must really mean missing
                      ]

    # This option is appropriate for attributes where NA signifies something meaningful - not just a missing value
    replaceWithNotApp = ['Alley',         # NA means "no alley access" C
                         'Utilities',     # NA means "no utilities" 
                         'BsmtQual',      # NA means "no basement" C
                         'BsmtCond',      # NA means "no basement" C
                         'BsmtExposure',  # NA means "no basement" C
                         'BsmtFinType1',  # NA means "no basement" C
                         'BsmtFinType2',  # NA means "no basement" C
                         'FireplaceQu',   # NA means "no fireplace" C
                         'GarageType',    # NA means "no garage" C
                         'GarageFinish',  # NA means "no garage" C
                         'GarageQual',    # NA means "no garage" C
                         'GarageCond',    # NA means "no garage" C
                         'PoolQC',        # NA means "no pool"  C
                         'Fence',         # NA means "no fence" C
                         'MiscFeature'    # NA means "none" - no additional features
                        ]

    for attr in replaceWithZero:
        allInputs[attr] = allInputs[attr].fillna(0)
        
    for attr in replaceWithMode:
        allInputs[attr] = allInputs[attr].fillna(allInputs[attr].mode().iloc[0])

    for attr in replaceWithMean:
        allInputs[attr] = allInputs[attr].fillna(allInputs[attr].mean())

    for attr in replaceWithNotApp:
        allInputs[attr] = allInputs[attr].fillna('NotApp')

    # GarageYrBlt - NA means there is no garage. We could make this NotApp, but then we don't consistently have a year type anymore.
    # So we'll replace with the year the house was built.
    attr = 'GarageYrBlt'  # C
    allInputs.loc[np.isnan(allInputs[attr]), attr] = allInputs.loc[np.isnan(allInputs[attr]), 'YearBuilt']

def transformData(trainDF, testDF):
    # ----------------------------------------------------------------------
    # Set up the predictors
    predictors = getPredictors(trainDF)

    # -----------------------------------------------------------------------
    # Gather the data

    trainOutputs = trainDF['SalePrice']

    # https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models/comments
    # All inputs except ID and SalePrice, across training and testing set
    allInputs = pd.concat((trainDF.loc[:, 'MSSubClass':'SaleCondition'],
                           testDF.loc[:, 'MSSubClass':'SaleCondition']))
    # Using allInputs, we'll transform all data at once in a consistent way

    # ------------------------------------------------------------------------
    # Determine replacement policy for missing values

    #missing = getAttrsWithMissingValues(allInputs)

    __handleMissingAttributes(allInputs)

    #id = 235
    #row = id-1
    #print("row", row, ":", allInputs.iloc[row][missing])
    #print("row", row, ":", allInputs.iloc[row])

    #testID = 1916
    #row = trainDF.shape[0] + (testID-1461) # 1461 is the id of the first test case
    #print("testID", testID, ":", allInputs.iloc[row])

    # -----------------------------------------------------------------------
    # Convert year types into years since (years since built, years since remodeled, etc.)

    yearAttrs = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
    currentYear = 2016
    for attr in yearAttrs:
        allInputs[attr] = currentYear - allInputs[attr]

    #print(allInputs.iloc[0][yearAttrs])

    # -----------------------------------------------------------------------
    # Do a log transform on skewed numeric attributes

    numericAttrs = getNumericAttrs(allInputs)

    # https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models/comments
    # Using only the trainDF, not allInputs, so that training is only based on training data
    # Drop missing values temporarily when determining amount of skew
    skewedAttrs = trainDF[numericAttrs].apply(lambda x: scipy.stats.skew(x.dropna()))  # A Series showing skew for each attribute
    skewedAttrs = skewedAttrs[skewedAttrs > 0.75]                          # Just the attributes that are "highly skewed"
    #print(skewedAttrs.values)       # The values that occur in the series
    #print(skewedAttrs.index)        # The attributes that occur in the series
    skewedAttrs = skewedAttrs.index

    # https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models/comments
    # Do a log transform on all attributes that are skewed
    allInputs[skewedAttrs] = np.log1p(allInputs[skewedAttrs])

    # ------------------------------------------------------------------------
    # Convert all numeric attributes to a [0,1] range
    for attr in numericAttrs:
        allInputs[attr] = allInputs[attr] / allInputs[attr].max()

    # printTopN(allInputs, 5, allInputs.columns)

    # ------------------------------------------------------------------------
    # Convert Neighborhood to 0-1 range sorted by average price

    attrToAverages = allAveragePricesByAttrValue(allInputs, allInputs[:trainDF.shape[0]], trainOutputs)
    attr = 'Neighborhood'
    convertAndNormalizeByAvgPrice(allInputs, attr, attrToAverages[attr])
    allInputs['Neighborhood'] = allInputs['Neighborhood'].astype(float)  # Make sure the numbers are treated as floats, not objects

    # ------------------------------------------------------------------------
    # Convert remaining non-numeric attributes to separate binary attributes

    #nonNumericAttrs = getNonNumericAttrs(allInputs)
    allInputs = pd.get_dummies(allInputs)

    # Since some predictors may have now been changed to multiple binary attributes, we
    # need to change the predictors list accordingly
    
    newPredictors = []
    for origPred in predictors:
        containingAttrs = [attr for attr in allInputs.columns if origPred == attr[:len(origPred)]]
        newPredictors = newPredictors + containingAttrs
    
    predictors = newPredictors
    
    #printTopN(allInputs, 5, allInputs.columns)    

    return(allInputs[:trainDF.shape[0]], allInputs[trainDF.shape[0]:], trainOutputs, testDF['Id'], predictors)


# ============================================================================
# Set up the predictors

'''
Set what predictors will be used in the algorithm.
'''


def getPredictors(trainDF):
    # Option 1: Specify the predictors you want
    predictors = ['HouseStyle','Alley','TotRmsAbvGrd','CentralAir','BsmtExposure','BsmtCond','BsmtQual','ExterQual','1stFlrSF','BsmtFullBath','GarageQual','SaleType','SaleCondition','GrLivArea','TotalBsmtSF','MSSubClass', 'LotArea', 'FullBath', 'BedroomAbvGr', 'MSZoning', 'Neighborhood', 'BldgType',
                  'OverallQual', 'OverallCond', 'KitchenQual', 'YearBuilt', 'YearRemodAdd']

    # Additional added attributes that improved the results - 'SaleType','SaleCondition','GarageQual','BsmtFullBath','1stFlrSF','CentralAir','BsmtExposure','BsmtCond','BsmtQual','GrLivArea','HouseStyle','TotalBsmtSF', 'Alley', 'TotRmsAbvGrd','ExterQual'
    
    # Checked scores by adding these attributes one by one but didn't make significant/any difference - 
    # 'KitchenAbvGr','MiscVal','Fence','PoolArea','GarageCond','HeatingQC','Fireplaces', 'FireplaceQu','Street','LotShape','KitchenQual','ExterCond' 


    # Option 2: Specify the predictors you don't want
    # all = [p for p in list(trainDF.columns.values) if p not in ['Id', 'SalePrice']] # We should always remove these two.
    # remove = [] # Put what else should not be a predictor here.
    # predictors = [p for p in all if p not in remove]

    return predictors


# ============================================================================
# Converting values and normalizing by price

'''
Just a simple helper function to determine if two lists are equal or not.
'''


def __listEquals(a, b):
    if (len(a) != len(b)):
        return False

    for i in range(len(a)):
        if (a[i] != b[i]):
            return False
    return True


'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
This is also the place to add in hard-coding of attribute values that appear in the test set but not in the training set.
'''
def __getAttrToValuesDictionary(allInputs):
    attrToValues = {}
    for attr in allInputs.columns.values:
        values = sorted(allInputs[attr].unique())
        attrToValues[attr] = values

    return (attrToValues)


'''
A helper method for allAveragePricesByAttrValue - probably doesn't need to be called directly.
Returns a list of tuples (attr value, avg SalePrice) sorted from lowest avg SalePrice to highest
'''
def __averagePriceByAttrValue(trainIO, attr, values):
    globalMeanPrice = trainIO['SalePrice'].mean()

    averages = []
    for v in values:
        pricesOfRowsWithValueV = trainIO.loc[trainIO[attr] == v, 'SalePrice']
        if (pricesOfRowsWithValueV.shape[0] == 0):  # Then this is an attribute value for which we have no examples in the training set
            averages.append((v, globalMeanPrice))  # Just use the average sale price across the entire training set
        else:
            averages.append((v, pricesOfRowsWithValueV.mean()))  # Use the average sale price among rows with value v for the current attribute

    averages = sorted(averages, key=lambda pair: pair[1])  # The key to sort on is element [1] in the tuple
    # Now, averages is a sorted list of tuples (attr value, avgSalePrice)

    return (averages)


'''
Returns a dictionary mapping attribute to averages structure.
An averages structure is a sorted list of tuples (attr value, avg SalePrice)
This needs to be called on the training data, since the testing data doesn't have a SalePrice attribute
'''
def allAveragePricesByAttrValue(allInputs, trainInputs, trainOutputs):
    trainIO = pd.concat([trainInputs, trainOutputs], axis=1)

    attrToValues = __getAttrToValuesDictionary(allInputs)
    attrToAverages = {}
    for attr in allInputs.columns.values:
        attrToAverages[attr] = __averagePriceByAttrValue(trainIO, attr, attrToValues[attr])

    return attrToAverages


'''
Given the averages computed from the training set, converts and normalizes the given df (either trainDF or testDF)
in the order specified by those averages.
For example, if df is the testDF, attr is Neighborhood, and averages is a list of tuples (Neighborhood, avg SalePrice)
sorted by avg SalePrice, it normalizes the Neighborhood column in testDF into a [0,1] range, where 0 is the Neighborhood
with the lowest average SalePrice (in *trainDF*, not testDF, since we don't know SalePrices in testDF), 1 is the
Neighborhood with the highest average SalePrice, and all other neighborhoods are ordered in the middle.
In this way, the normalized [0,1] values for Neighborhood reflect a lot of information about expected sale price.
'''
def convertAndNormalizeByAvgPrice(df, attr, averages):
    # Find the sum of the average prices
    total = 0
    for pair in averages:
        total += pair[1]

    # Reassign each non-numeric value proportional to its average price
    values = df[attr].unique()  # values is an array of the possible values for the current attribute
    numValues = values.size
    sumSoFar = 0
    for attrVal, price in averages:
        sumSoFar += price
        df.loc[df[attr] == attrVal, attr] = sumSoFar / total


'''
Mutates df by converting and normalizing the values for attr.
The set of unique values for attr is obtained, ordered arbitrarily,
and mapped to the range [0,1].
'''


def convertAndNormalizeUnordered(df, attr):
    # The unique() method returns an array of the unique values in the object
    values = df[attr].unique()  # values is an array of the possible values for the current attribute
    numValues = values.size
    replacementVal = 0  # The current numerical replacement value
    for v in values:
        df.loc[df[attr] == v, attr] = replacementVal / (numValues - 1)
        replacementVal += 1

# Use this normalization for all the Continuous attributes

def convertAndNormalizeByMax(df,attr):
    for i in attr:
        df[attr] = df[attr] / df[attr].max()


'''
This is the function to "hardcode" the conversion and normalization procedure for each attribute.
Options so far are:

1) Use convertAndNormalizeByAvgPrice - If you do this approach, don't do any other modifications of that attribute (-1, log, etc.) since
the averages are based on the original attr values.
2) Use convertAndNormalizeUnordered
3) Use simple calculations involving .max()
4) use df[attr] = pd.get_dummies(df[attr]) to replace the attribute with multiple binary columns

Utkarsh's notes:
Normalize all the continuous attributes via method no. 3
Normalize all the nominal & ordinal attributes via method no. 1
Discrete variables can be normalized either way (3 Would be better in my opinion since we could miss some values while normalizing by method 1)

Note: Maybe we shouldn't adjust how we handle each attribute unless the change makes a "significant" difference. Very small differences may
just be overfitting the training data and not lead to better generalization in the testing data. We should go with what seems to make the
most sense.
'''


# ==========================================================================
# Testing functions

def rootMeanSquareLogError(pred, target):
    return (np.sqrt(np.square(np.log(pred + 1) - np.log(target + 1)).mean()))

'''
Give 'r2' or 'rmse' for the scoring method
'''
def doCrossValidation(inputs, outputs, alg, scoring='r2'):
    if (scoring=='r2'):
        scores = model_selection.cross_val_score(alg, inputs, outputs, cv=3, scoring=scoring)

    elif (scoring=='rmse'):
        scores = np.sqrt(-model_selection.cross_val_score(alg, inputs, outputs, cv=3, scoring="neg_mean_squared_error"))

    else:
        scores = [0]

    return (scores.mean())


# ===============================================================================
# What missing values exist?

def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis='index')  # 'index' means to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (
    numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].axes[0]  # similar to computing nonNumericAttrs, above
    return (attrsWithMissingValues)

# =============================================================================
# =============================================================================

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric[isNumeric==findNumeric] # selects the values in isNumeric that are <findNumeric> (True or False)
    attrs = attrs.axes # a list of the row axis labels: the (non)numeric attribute labels
    attrs = attrs[0] # The above is actually a singleton list, so this gets inside of it
    return(attrs)

def getNumericAttrs(df):
    return(__getNumericHelper(df, True))

def getNonNumericAttrs(df):
    return(__getNumericHelper(df, False))


'''
Mutates the given dataframe by replacing all non-numeric values with numeric ones starting at 0.
'''
# Set normalize to True to normalize
# def replaceNonNumeric(df, normalize):
#    # Find non-numeric attributes
#    nonNumericAttrs = getNonNumericAttrs(df)

#    # Replace non-numeric attributes
#    for attr in nonNumericAttrs:
#        replaceNonNumericAttr(df, attr, normalize)

# =============================================================================
# PCA - doesn't work yet

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
#%matplotlib inline

def doPCA():
    #Load data set
    data = pd.read_csv('mod_train.csv')

    #convert it to numpy arrays
    X=data.values

    #Scaling the values
    X = scale(X)
    
    pca = PCA(n_components=50)
    
    pca.fit(X)
    
    #The amount of variance that each PC explains
    var= pca.explained_variance_ratio_
    
    #Cumulative Variance explains
    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    
    print (var1)
    
    #plt.plot(var1)
    
    pca = PCA(n_components=50)
    pca.fit(X)
    X1=pca.fit_transform(X)
    
    print (X1)

# =============================================================================

if __name__ == "__main__":
    main()

# Best results obtained by removing 'TotalBsmtSF' attribute & running linear regression. (0.19)
# on adding 'TotalBsmtSF' attribute quantile regression gave better result(0.20) than linear reg(0.21).
