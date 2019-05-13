from sklearn import neighbors, datasets
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap

def dist1(d):
    a= 0.1 #change a from 0.1,1,10,100 and try it!
    return np.exp((-a)*np.power(d,2))
    
def dist2(d):
    a= 10 #change a from 0.1,1,10,100 and try it!
    return np.exp((-a)*np.power(d,2))
    
def dist3(d):
    a= 100 #change a from 0.1,1,10,100 and try it!
    return np.exp((-a)*np.power(d,2))
    
def dist4(d):
    a= 1000 #change a from 0.1,1,10,100 and try it!
    return np.exp((-a)*np.power(d,2))

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_normalized = preprocessing.scale(X)
X_t = PCA(2).fit_transform (X_normalized)
X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.4)
model = GaussianNB()
h=0.01
x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

for n_neighbors in range (1,11):
    
    clf = neighbors.KNeighborsClassifier(n_neighbors, metric = 'euclidean' )
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.title('%d n_neighbors map without any weight' %n_neighbors)
    plt.scatter(X_t[:,0],X_t[:, 1],c=y)
    plt.show()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print("The accuracy of the %d. is: %f \n" % (n_neighbors, clf.score(X_test, y_test)))
    
print("============== End of for loop! ================\n")

print("=======================The distance weight part===============================\n")


n_neighbors = 3
clf = neighbors.KNeighborsClassifier(n_neighbors, weights = 'distance', metric = 'euclidean' )
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.title('%d n_neighbors map distance weight' %n_neighbors)
plt.scatter(X_t[:,0],X_t[:, 1],c=y)
plt.show()
y_pred = model.fit(X_train, y_train).predict(X_test)
print("The accuracy of the %d. is: %f \n" % (n_neighbors, clf.score(X_test, y_test)))

print("=======================The uniform weight part===============================\n")

clf = neighbors.KNeighborsClassifier(n_neighbors, weights = 'uniform', metric = 'euclidean' )
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.title('%d n_neighbors map uniform weight' %n_neighbors)
plt.scatter(X_t[:,0],X_t[:, 1],c=y)
plt.show()
y_pred = model.fit(X_train, y_train).predict(X_test)
print("The accuracy of the %d. is: %f \n" % (n_neighbors, clf.score(X_test, y_test)))

print("=======================Your gaussian function as weight===============================\n")

print("=======================Alfa is 0.1 now!!!===============================\n")
clf = neighbors.KNeighborsClassifier(n_neighbors, weights = dist1, metric = 'euclidean' )
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.title('%d n_neighbors map alfa is 0.1' %n_neighbors)
plt.scatter(X_t[:,0],X_t[:, 1],c=y)
plt.show()
y_pred = model.fit(X_train, y_train).predict(X_test)
print("The accuracy of the %d. is: %f \n" % (n_neighbors, clf.score(X_test, y_test)))
        
print("=======================Alfa is 10 now!!!===============================\n")
clf = neighbors.KNeighborsClassifier(n_neighbors, weights = dist2, metric = 'euclidean' )
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.title('%d n_neighbors map alfa is 10' %n_neighbors)
plt.scatter(X_t[:,0],X_t[:, 1],c=y)
plt.show()
y_pred = model.fit(X_train, y_train).predict(X_test)
print("The accuracy of the %d. is: %f \n" % (n_neighbors, clf.score(X_test, y_test)))

print("=======================Alfa is 100 now!!!===============================\n")
clf = neighbors.KNeighborsClassifier(n_neighbors, weights = dist3, metric = 'euclidean' )
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.title('%d n_neighbors map alfa is 100' %n_neighbors)
plt.scatter(X_t[:,0],X_t[:, 1],c=y)
plt.show()
y_pred = model.fit(X_train, y_train).predict(X_test)
print("The accuracy of the %d. is: %f \n" % (n_neighbors, clf.score(X_test, y_test)))
    
print("=======================Alfa is 1000 now!!!===============================\n")
clf = neighbors.KNeighborsClassifier(n_neighbors, weights = dist4, metric = 'euclidean' )
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.title('%d n_neighbors map alfa is 1000' %n_neighbors)
plt.scatter(X_t[:,0],X_t[:, 1],c=y)
plt.show()
y_pred = model.fit(X_train, y_train).predict(X_test)
print("The accuracy of the %d. is: %f \n" % (n_neighbors, clf.score(X_test, y_test)))

for n_neighbors in range (1,11):

    print("=======================Alfa now!!!===============================\n")
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights = dist1, metric = 'euclidean' )
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.title('%d n_neighbors map alfa is 0.1' %n_neighbors)
    plt.scatter(X_t[:,0],X_t[:, 1],c=y)
    plt.show()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print("The accuracy of the %d. is: %f \n" % (n_neighbors, clf.score(X_test, y_test)))
   
for n_neighbors in range (1,11):
    print("Weight is distance now!!!!\n")
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights = 'distance', metric = 'euclidean' )
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.title('%d n_neighbors map distance weight' %n_neighbors)
    plt.scatter(X_t[:,0],X_t[:, 1],c=y)
    plt.show()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print("The accuracy of the %d. is: %f \n" % (n_neighbors, clf.score(X_test, y_test)))

for n_neighbors in range (1,11):
    print("Weight is uniform now!!!!\n")
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights = 'uniform', metric = 'euclidean' )
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.title('%d n_neighbors map uniform weight' %n_neighbors)
    plt.scatter(X_t[:,0],X_t[:, 1],c=y)
    plt.show()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print("The accuracy of the %d. is: %f \n" % (n_neighbors, clf.score(X_test, y_test)))
