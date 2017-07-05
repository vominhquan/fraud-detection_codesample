import pandas as pd
import numpy as np
from sklearn import svm
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss

data = pd.read_csv('creditcard.csv')

data = data.values

print np.shape(data)

label = data[:,30]

print np.shape(label)

data = data[:,:30]

print np.shape(data)

#nm = NearMiss(version=3, return_indices=True) 
#
#data_resampled, label_resampled, idx_resampled = nm.fit_sample(data, label)

cc = ClusterCentroids(ratio=0.4)
data_resampled, label_resampled = cc.fit_sample(data, label)

X_train = data_resampled[492:]
X_test = data_resampled[:492]
#
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
#
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
#
n_error_train = y_pred_train[y_pred_train == -1].size
rec = y_pred_test[y_pred_test == -1].size/492
#
print n_error_train
print rec
