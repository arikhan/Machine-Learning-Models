import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


Xtest = np.load('\\Users\\PC\\Desktop\\regression_Xtest.npy')
Xtrain = np.load('\\Users\\PC\\Desktop\\regression_Xtrain.npy')
ytest = np.load('\\Users\\PC\\Desktop\\regression_ytest.npy')
ytrain = np.load('\\Users\\PC\\Desktop\\regression_ytrain.npy')
arrayForMean = []
#
#lr = LinearRegression()
#lr.fit( x_train.reshape(-1,1), y) #//LR.fit() wants an array
#plt.plot(X_test, lr.predict(X_test.reshape(-1,1), label="Model")
#plt.scatter(Xtrain.reshape(-1,1), ytest, c='r')
#predicted = lr.predict(X_test.reshape(-1,1))
##mean_square_error = … //you will need x_test and y_test

for k in range (1,10):
    
    poly = PolynomialFeatures(degree=k, include_bias=False)
    xPoly = poly.fit_transform(Xtrain.reshape(-1,1))
    lr = LinearRegression()
    lr.fit(xPoly, ytrain)
    x_range = np.linspace(-1, 5.5, 100)
    predicted = lr.predict(poly.fit_transform(x_range.reshape(-1,1)))
    #predicted ones are the lines 
    plt.plot(x_range.reshape(-1,1), predicted)
    plt.scatter(Xtest.reshape(-1,1), ytest, c='r')
    plt.show()
    predicted2 = lr.predict(poly.fit_transform(Xtest.reshape(-1,1)))
    mean_square_error = mean_squared_error(ytest, predicted2)
    arrayForMean.append(mean_square_error)
    print(" %d. mean square error is: %.5f \n" % (k,mean_square_error))
    
plt.plot(np.arange(1,10),arrayForMean)