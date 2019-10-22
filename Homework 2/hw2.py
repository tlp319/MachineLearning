# Luke Pearson
# HW2
# 10/3/2019

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


class K_Accuracy:
  def __init__(self, k_val, accuracy, returnArray):
    self.k_val = k_val
    self.accuracy = accuracy
    self.returnArray = returnArray


def k_means(k):
    digits = load_digits()
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(digits.data)
    labels = np.zeros_like(clusters)
    
    for i in range(k):
        mask = (clusters == i)
        labels[mask] = mode(digits.target[mask])[0]

    return [str(accuracy_score(digits.target, labels)), labels, digits]




def main():
    # Range of K values to test = 1 to ks
    ks = 1000
    bestK = K_Accuracy(0, 0, None)
    for i in range(ks):
        currentK = K_Accuracy(i+1, k_means(i+1)[0], None)
        # Uncomment for debugging
        # print "K = " + str(currentK.k_val) + " has accuracy: " + str(currentK.accuracy)
        if currentK.accuracy > bestK.accuracy:
            bestK = currentK

    bestK.returnArray = k_means(bestK.k_val)
            
    print "For K's in range 1 to " + str(ks) + ":\nthe best K for k-means is " + str(bestK.k_val) + " with an accuracy of " + str(bestK.accuracy)
    
    mat = confusion_matrix(bestK.returnArray[2].target, bestK.returnArray[1])
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=bestK.returnArray[2].target_names, yticklabels=bestK.returnArray[2].target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()
        
main()
