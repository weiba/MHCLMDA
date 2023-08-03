import numpy as np

def Jaccard_sim(x):
    """
    Calculate the Jaccard similarity between each pair of rows in x
    :param x: N X D
                N: the number of objects
                D: Dimension of the feature
    :return: N X N similarity matrix
    """
    x = np.array(x)
    intersection = np.dot(x, x.T)
    union = np.sum(x, axis=1, keepdims=True) + np.sum(x, axis=1, keepdims=True).T - intersection
    similarity_mat = intersection / union
    return similarity_mat

# DG = np.loadtxt('data/dg_delete.txt')#435*11216
# MG = np.loadtxt('data/mg_delete.txt')#757*11216
# mg = Jaccard_sim(MG)
# dg = Jaccard_sim(DG)

MD = np.loadtxt('data/dg_delete.txt')#435*11216
MG = np.loadtxt('data/mg_delete.txt')#757*11216
mg = Jaccard_sim(MG)
dg = Jaccard_sim(DG)

len(X.shape)

