# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Most ML algorithm are based on euclidean distance. 
# If features are on different scale (age vs salary's range)
# then euclidean space will be dominated by salary.
# Tho even for decision trees, we should still scale so that
# the algorithm can converge much faster. 

# Standardization
# x-mean/std
# Normalization
# x-min(x)/max(x)-min(x)
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# We are also scaling the dummy vars. 
# If we scale, everything will be on the same scale. 
# but we lose the intuition. 
X_train = sc_X.fit_transform(X_train)
# No fit on test set. 
X_test = sc_X.transform(X_test)
# No need to fit Y set because this is a classification problem.\
"""