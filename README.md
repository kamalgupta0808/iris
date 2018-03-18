# iris

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

dataset=pandas.read_csv('/home/kamal/ML/Datasets/Iris.csv')
dataset.head(5)

dataset.shape
dataset.describe()
print(dataset.groupby('Species').size())
dataset[dataset['Species']=='Iris-setosa'].describe()

dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
# iris = sns.load_dataset("iris")
dataset["ID"] = dataset["Id"]
dataset["ratio"] = dataset["SepalLengthCm"]/dataset["SepalWidthCm"]

sns.lmplot(x="ID", y="ratio", data=dataset, hue="Species", fit_reg=False, legend=False)

plt.legend()
plt.show()

dataset["ID"] = dataset["Id"]
dataset["ratio"] = dataset["PetalLengthCm"]/dataset["PetalWidthCm"]

sns.lmplot(x="ID", y="ratio", data=dataset, hue="Species", fit_reg=False, legend=False)

plt.legend()
plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,1:5]
print (X)
Y = array[:,5]
print (Y)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
  
 
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print (X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

import numpy as np
a=np.array([[3.8,1,4,1.5]])
print (a.shape)
prediction = knn.predict(a)
print (prediction)
