from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

BestAccuracy = 0
BestAccuracy1 = 0
Cset = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
gammas = [10**-9,10**-7,10**-5,10**-3,10**-1]
accuracyLinear = []
accuracyRbf = []
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_first_Two = X[:, 0:2]
#print(X_first_Two)
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(X_first_Two, y, test_size=.5)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.6)

h = 0.01
#1/3 ün 4. sorusu burada
for w in Cset:
    svc = svm.SVC(kernel='linear', C=w).fit(x_train, y_train)
    rbf_svc = svm.SVC(kernel='rbf', gamma='auto', C=w).fit(x_train, y_train)
    x_min, x_max = X_first_Two[:, 0].min() - 1, X_first_Two[:, 0].max() + 1
    y_min, y_max = X_first_Two[:, 1].min() - 1, X_first_Two[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    
    titles = ['SVC with linear kernel',
              'SVC with RBF kernel']
#plot for both linear and rbf kernel
    for i, clf in enumerate((svc, rbf_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        
    #    # Plot also the training points
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap = plt.cm.Paired)
        plt.xlabel('length')
        plt.ylabel('width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
        
        plt.show()
        score = clf.score(x_validation, y_validation)
        print("The accuracy is:%.5f" % (score))
        if (i==0):
            accuracyLinear.append(score)
            if(BestAccuracy <= score):
                BestAccuracy = score
                C=w
                score1 = clf.score(x_test, y_test)
        elif(i==1):
            accuracyRbf.append(score)
            if(BestAccuracy1 <= score):
                BestAccuracy1 = score
                C1=w
                score2 = clf.score(x_test, y_test)
            
#accuracy data for linear kernel plotted 
plt.plot(accuracyLinear)
plt.axis([0, 6, 0, 1])
plt.title('Accuracy on Validation on Linear SVM')
plt.show()
print("The best C is:%f" % (C))
print("The accuracy of the Linear SVM is:%.5f" % (BestAccuracy))
print("The accuracy of the test set is :%.5f" % (score1))
print(accuracyLinear)
#accuracy data for RBF kernel plotted
plt.plot(accuracyRbf)
plt.axis([0, 6, 0, 1])
plt.title('Accuracy on Validation on RBF SVM')
plt.show()
print("The best C for RBF is:%f" % (C1))
print("The accuracy of the RBF SVM is:%.5f" % (BestAccuracy1))
print("The accuracy of the test set is :%.5f" % (score2))
print(accuracyRbf)

#Grid search part
param_grid = dict(gamma=gammas, C=Cset)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid).fit(x_validation, y_validation)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

grid.fit(x_validation, y_validation)

for params, mean_score, scores in grid.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
bestparams = {}
bestparams = grid.best_params_
rbf_svc = svm.SVC(kernel='rbf', gamma=bestparams['gamma'], C=bestparams['C']).fit(x_train, y_train)
#plot the best gamma and C values for RBF
plt.subplot(2, 2, 0 + 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# Plot also the training points
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap = plt.cm.Paired)
plt.xlabel('length')
plt.ylabel('width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(titles[i])
        
plt.show()
print("The accuracy of test sets is: %0.5f" %(grid.score(x_test,y_test)))

#For K-Fold we know what to do. 
x_trainVal = np.concatenate((x_train, x_validation), axis=0)
y_trainVal = np.concatenate((y_train,y_validation), axis=0)

param_grid = dict(gamma=gammas, C=Cset)
grid = GridSearchCV(svm.SVC(kernel = 'rbf'), param_grid=param_grid, cv=5).fit(x_validation, y_validation)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

grid.fit(x_validation, y_validation)

for params, mean_score, scores in grid.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
bestparams = {}
bestparams = grid.best_params_
#plot the best gamma and C values for RBF
plt.subplot(2, 2, 0 + 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# Plot also the training points
plt.scatter(x_trainVal[:, 0], x_trainVal[:, 1], c=y_trainVal, cmap = plt.cm.Paired)
plt.xlabel('length')
plt.ylabel('width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(titles[i])
        
plt.show()
print("The accuracy of test sets is: %0.5f" %(grid.score(x_test,y_test)))
