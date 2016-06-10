%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from scipy.stats import pearsonr, normaltest
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import metrics
import warnings
import math
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import scipy.stats as stats


warnings.simplefilter(action = "ignore", category = FutureWarning)

####Part 1
#Import the data and merge tables

airportsdf = pd.read_csv('/Users/Starshine/DSI/Week-8/Project-7/assets/airports.csv')
cancelsdf = pd.read_csv('/Users/Starshine/DSI/Week-8/Project-7/assets/airport_cancellations.csv')
opsdf = pd.read_csv('/Users/Starshine/DSI/Week-8/Project-7/assets/Airport_operations.csv')

airportsdf.shape
cancelsdf.shape
opsdf.shape

#####Part 2: Plot and describe the data
#I chose to plot histograms of the operations dataframe columns
X = opsdf.iloc[:, 2:15]
for col in X.columns.values:
    sns.distplot(X[col])
    plt.show()

#The data looks about normally distributed

#####Part 3
#1. Create dummies/label encode the airports for use later in PCA analysis
le = LabelEncoder()
y = opsdf.airport
y = le.fit_transform(y)

#####Part 4
opsdf
#normal distribution test
stats.normaltest(X, axis=0)
#we see that the p values are way less than .05, so we can infer each column of data is NOT normally distributed.
#normaltest returns a 2-tuple of the chi-squared statistic, and the associated p-value. Given the null hypothesis that x came from a normal distribution,
#the p-value represents the probability that a chi-squared statistic that large (or larger) would be seen.
#If the p-val is very small, it means it is unlikely that the data came from a normal distribution.

#checkout the correlations for each feature
X.corr()

####Part 5

opsdf.head(3)


#set the x and y
x = opsdf.ix[:,2:14].values
y = opsdf.ix[:,0].values

#standardize the x variable for analysis
xStand = StandardScaler().fit_transform(x)
#Next, create the covariance matrix from the standardized x-values and decompose these values to find the eigenvalues and eigenvectors
covariancematrix = np.cov(xStand.T)
eigenValues, eigenVectors = np.linalg.eig(covariancematrix)

print eigenValues
print eigenVectors

#To find the principal components, find the eigenpairs, and sort them from highest to lowest.
eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]
eigenPairs.sort()
eigenPairs.reverse()
for i in eigenPairs:
    print(i[0])

#Next, calculate the explained variance
totalEigen = sum(eigenValues)
varExpl = [(i / totalEigen)*100 for i in sorted(eigenValues, reverse=True)]
cumulvarExpl = np.cumsum(varExpl)

print cumulvarExpl
#Here we see that close to 93% of the variance is explained by the first 4 principal components

#perform the PCA
pca = PCA(n_components =2)
Y = pca.fit_transform(xStand)

#make Y into a dataframe
Ydf= pd.DataFrame(Y)


#create new dataframe with airport and year and join with the PCA dataframe
airportyeardf = opsdf[['airport', 'year']]

air_pca = airportyeardf.join(Ydf, on=None, how='left')
air_pca

air_pca.columns = ['airport', 'year', 'PC1', 'PC2']


#now graph the results in scatter plot form
graph = air_pca.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8))
for i, airport, in enumerate(opsdf['airport']):
    graph.annotate(airport, (air_pca.iloc[i].PC2, air_pca.iloc[i].PC1))


#Summary/Conclusions
'''Our visualization of PCA1 and PCA2 shows us that both TEB and VNY airports are outliers in our dataset.
Most airports cluster near (.5, -1), with ATL representing the opposite outlier in our dataset. ATL could
stand to do better with delays, though we do know from our analysis it handles much more traffic than our
better performing airports. Generally speaking, our data is not normally distributed as seen in our normality
test. It doesn't come as a surprise to see strong correlations (.89+) between on-time gate departures and
on-time departures and on-time arrivals. I would look into a faster taxiing system for arrivals considering
their effect on gate-departures.'''
