
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# create the Spark Session
spark = SparkSession.builder.getOrCreate()

# create the Spark Context
sc = spark.sparkContext

# convert string into array
def rowProcessor(x):
  return np.array([float(i) for i in x.split(' ')])

# import data and initial centroids
data = sc.textFile('kmeans_data.txt').map(rowProcessor)
c1 = sc.textFile('c1.txt').map(rowProcessor)
c2 = sc.textFile('c2.txt').map(rowProcessor)

# convert centroids into ndarray
c1, c2 = np.array(c1.take(c1.count())), np.array(c2.take(c2.count()))

# find the cluster associated with the least L2 distance
def findClustersL2(x, centroids, k):
  clusterInd = 0
  cloestDist = np.inf
  for i in range(k):
    EucDist = np.sum((x-centroids[i]) **2)
    if EucDist < cloestDist:
      cloestDist = EucDist
      clusterInd = i
  return dict([("cluster",clusterInd), ("distance", cloestDist) ])

# k-means
def kMeansIter(data, nIter, initialCentroids, k, d):
  centroids = initialCentroids
  cost = np.sum(np.array(data.map(lambda x: findClustersL2(x, centroids, k)['distance']).collect()))
  for i in range(nIter):
    #update centroids
    clusterLabels = np.array(data.map(lambda x: findClustersL2(x, centroids, k)['cluster']).collect())
    centroids = np.zeros((k, d))
    for j in range(k):
        centroids[j,:] = np.mean(np.array(data.collect())[np.where(clusterLabels==j)], axis=0)
    
    #calculate and store the cost
    c = np.sum(np.array(data.map(lambda x: findClustersL2(x, centroids, k)['distance']).collect()))
    cost = np.append(cost, c)
  d = dict([('cost',cost), ('centroids', centroids)])
  return d

# perform k-means 
k, d = c1.shape[0], c1.shape[1]
kmeansC1 = kMeansIter(data, 20, c1, k, d)
kmeansC2 = kMeansIter(data, 20, c2, k, d)

# get predicted clusters
kmeansC1_labels = data.map(lambda x: findClustersL2(x, kmeansC1['centroids'], k)['cluster']).collect()
kmeansC2_labels = data.map(lambda x: findClustersL2(x, kmeansC2['centroids'], k)['cluster']).collect()

# plot cost per iteration
x = [str(i) for i in range(21)]
plt.scatter(x=x, y=kmeansC1['cost'], label='c1')
plt.scatter(x=x, y=kmeansC2['cost'], label='c2')
plt.legend(loc='upper right')
plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("cost of k-means")
plt.show()

# get percentage changes after the 10th interation
print('The cost of k-means increased by',(kmeansC1['cost'][10]-kmeansC1['cost'][1])/kmeansC1['cost'][1] * 100, 
      'percent', '\n', 'after 10 iterations with c1 for initialization')
print('The cost of k-means increased by',(kmeansC2['cost'][10]-kmeansC2['cost'][1])/kmeansC2['cost'][1] * 100, 
      'percent', '\n', 'after 10 iterations with c2 for initialization')