# Check Python version and Scikit-Learn version and update if necessary
import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import pandas as pd


#Importing packages for the ML methods

from sklearn.model_selection import train_test_split # Dataset splitter function
from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression    # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error    # function to calculate mean squared error 
from sklearn.svm import SVC     #SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hinge_loss

# To plot pretty figures

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Save data in dataframe 'df' and drop unused features. We will only use the features titled "name", "category" and "price"
# so we drop "currency" ( all of the data is in the same currency, $) and "description".
data = pd.read_csv('scotch_review.csv')
data = data.drop(['currency', 'description'], axis=1)

print('Total datapoints')
print(len(data.price))
#print(data.info)

data = data[data['price'] != '$15,000 or $60,000/set']
data['price'] = data['price'].str.replace("," ,"")
data['price'] = data['price'].str.replace("/set" ,"")
data['price'] = data['price'].str.replace("/liter" ,"")


data['price'] = data['price'].astype(float)

data = data.drop(data[data.price > 800].index)


print('Datapoints after cleaning:')
print(len(data.price))

# Let's say we want to split the data in 80:10:10 for train:valid:test dataset
train_size=0.8

X = data


y = data['review.point']

# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

print('Training set size')
print(len(X_train))
print('Validation set size')
print(len(X_valid.price))
print('Testing set size')
print(len(X_test.price))
# Making numpy arrays for each feature we are using to help out with the next steps and convert numbers that aren in string
# format to Int


#Training set
prices_train2 = np.array(X_train['price']).reshape(-1,1).astype(float)
prices_train = np.asarray(prices_train2, dtype=int)
scores_train2 = np.array(X_train['review.point']).reshape(-1,1).astype(float)
scores_train = np.asarray(scores_train2, dtype=int)


#Validation set
prices_val2 = np.array(X_valid['price']).reshape(-1,1).astype(float)
prices_val = np.asarray(prices_val2, dtype=int)
scores_val2 = np.array(X_valid['review.point']).reshape(-1,1).astype(float)
scores_val = np.asarray(scores_val2, dtype=int)

#Testing set
prices_test2 = np.array(X_test['price']).reshape(-1,1).astype(float)
prices_test = np.asarray(prices_test2, dtype=int).reshape(-1,1)
scores_test2 = np.array(X_test['review.point']).reshape(-1,1).astype(float)
scores_test = np.asarray(scores_test2, dtype=int)


# Fitting the linear regression model to the feature. 
price_regr = LinearRegression()
price_regr.fit(prices_train,scores_train)
y_pred = price_regr.predict(prices_train)
tr_error = mean_squared_error(scores_train, y_pred)
y_pred_val = price_regr.predict(prices_val)
tr_error_val = mean_squared_error(scores_val, y_pred_val)
print('Training error for Linear regression')
print(tr_error)
print('Validation error for Linear regression')
print(tr_error_val)




# Fitting the SVC model to the feature. 
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
scores_train_svc = scores_train.ravel() #Make 1D-array for SVC, compiler warns about 2D-arrays i.e np.reshape(-1,1)
clf.fit(prices_train, scores_train_svc)
clf_pred = clf.decision_function(prices_train)
clf_error = hinge_loss(scores_train, clf_pred)
print('Training error for SVC')
print(clf_error)
clf1 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
scores_val_svc = scores_val.ravel()
clf1.fit(prices_val, scores_val_svc)
clf_pred_val = clf1.decision_function(prices_val)
clf_error_val = hinge_loss(scores_val, clf_pred_val)
print('Validation error for SVC')
print(clf_error_val)


#Testing errors for both methods
y_pred_test = price_regr.predict(prices_test)
tr_error_test = mean_squared_error(scores_test, y_pred_test)

clf2 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
scores_test_svc = scores_test.ravel()
clf2.fit(prices_test, scores_test_svc)
clf_pred_test = clf2.decision_function(prices_test)
clf_error_test = hinge_loss(scores_test, clf_pred_test)

print('Testing error for Linear Regression:')
print(tr_error_test)
print('Testing error for SVC:')
print(clf_error_test)

plt.figure(figsize=(8, 8))    # create a new figure with size 8*8

# create a scatter plot of datapoints 
# each datapoint is depicted by a dot in color 'blue' and size '4'
plt.scatter(prices_test, scores_test, color='b', s=4) 
y_pred = price_regr.predict(prices_test)
# plot the predictions obtained by the learnt linear hypothesis using color 'red' and label the curve as "h(x)"
   # predict using the linear model
plt.plot(prices_test, y_pred, color='r')  
plt.xlabel('price',size=15) # define label for the horizontal axis 
plt.ylabel('score',size=15) # define label for the vertical axis 
plt.title('Linear regression model',size=15) # define the title of the plot, name of the used method  
plt.legend(loc='best',fontsize=14) # define the location of the legend  
plt.show()  # display the plot on the screen 

plt.figure(figsize=(8, 8))   
plt.scatter(prices_val, scores_val, color='b', s=8) 
plt.plot(prices_val, clf_pred_val, color='r')  
plt.xlabel('price',size=15) 
plt.ylabel('scores',size=15) 
plt.title('SVC model',size=15)  
plt.legend(loc='best',fontsize=14) 
plt.show()  


