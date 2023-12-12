from numpy import *
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import scipy.sparse as sp

def create_resultlist(result,testset,Index_PositiveRow,Index_PositiveCol,Index_zeroRow,Index_zeroCol,test_length_p,zero_length,test_f):

    result_list = zeros((test_length_p+len(test_f), 1))
    for i in range(test_length_p):
        result_list[i,0] = result[Index_PositiveRow[testset[i]], Index_PositiveCol[testset[i]]]
    for i in range(len(test_f)):
        result_list[i+test_length_p, 0] = result[Index_zeroRow[test_f[i]], Index_zeroCol[test_f[i]]]
    return result_list


def create_resultmatrix(result,testset,prolist):
    leave_col = prolist[testset]
    result = result[:,leave_col]
    return result

def create_resultmatrix_row(result,testset,prolist):
    leave_row = prolist[testset]
    result = result[leave_row, :]
    return result

def calculate_laplacian(adjacency_matrix):
    """
    Calculate the laplacian matrix for a given adjacency matrix.
    :param adjacency_matrix: a 2D numpy array or a scipy sparse matrix of shape (num_nodes, num_nodes).
    :return: a 2D numpy array or a scipy sparse matrix of shape (num_nodes, num_nodes), representing the laplacian matrix.
    """
    # Convert adjacency matrix to sparse format
    if sp.issparse(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocoo()

    # Calculate degree matrix
    degree_matrix = sp.diags(adjacency_matrix.sum(axis=1).flatten(), shape=adjacency_matrix.shape, dtype=np.double)
    # Calculate laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix
    laplacian_matrix = np.array(laplacian_matrix)


    return laplacian_matrix

