from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score

digits = load_digits()
X = digits.data
y = digits.target
X = X[y<5]
y = y[y<5]
N = 901 
purity = []
deneme = [] #################################################
clusterForGMM = [2,3,4,5,6,7,8,9,10]
NormalizedArray = []
HomogeneityArray = []
X_normalized = preprocessing.scale(X)
X_t = PCA(2).fit_transform (X_normalized)
kmeans = KMeans(n_clusters=5, random_state=0).fit(X_t)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(X_t[:, 0], X_t[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

clusters =[3,4,5,6,7,8,9,10]
for w in clusters:
    kmeans = KMeans(n_clusters=w, random_state=0).fit(X_t)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
    y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    plt.plot(X_t[:, 0], X_t[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('%d Clusters Graph for K-Means' %(w))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


    #GMM PART INCOMING!!!!!!!!!!!!!!!!!!!!!!!!!!
for w in clusterForGMM:
    GMM = sklearn.mixture.GMM(n_components=w
                                      ).fit(X_t)
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
    y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = GMM.predict(np.c_[xx.ravel(), yy.ravel()])
    yPredict = GMM.predict(X_t)
    deneme.append(yPredict)
    NormalizedArray.append(normalized_mutual_info_score(y , yPredict))
    HomogeneityArray.append(homogeneity_score(y , yPredict))
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    plt.plot(X_t[:, 0], X_t[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids2 = GMM.means_
    plt.scatter(centroids2[:, 0], centroids2[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('%d Clusters Graph for GMM' %(w))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    value = 0
    matrix = np.zeros((w,5))
    for cluster in range(0,w): 
        for label in range(0,5):
            for position in range(0,901):
                if (cluster == yPredict[position] and label == y[position]):
                    matrix[cluster][label] = matrix[cluster][label] + 1
        value = value + max(matrix[cluster])
    purity.append(value / N)
lines= plt.plot(clusterForGMM,purity, label="Purity")
lines1 = plt.plot(clusterForGMM,NormalizedArray, label="Normalized")
lines2 = plt.plot(clusterForGMM,HomogeneityArray, label="Homogeneity")
plt.setp(lines, color='r', linewidth=2.0)
plt.setp(lines1, color = 'b', linewidth=2.0)
plt.setp(lines2, color = 'g', linewidth = 2.0)
#plt.axis([2, 10, 0, 1])
plt.title('Score diagram')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


