from numpy import array
import numpy as np
from getData import getData
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler    
import matplotlib.pyplot as plt



class doPCA:
    """take pca"""
    def __init__(self) -> None:
        pass    
    def screePlot(features: array):
        pca = PCA(n_components=150)
        scl = StandardScaler()
        features = scl.fit_transform(features)
        pca.fit(features)
        plt.figure()
        plt.plot(pca.explained_variance_ratio_, 'o-', color='blue')
        plt.title('Scree Plot')
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Variance Explained")
        plt.show()
        plt.savefig("scree_plot.png")
        return None



if __name__ == '__main__':
    X,y = getData.loadFeatureMat()
    doPCA.screePlot(X)

