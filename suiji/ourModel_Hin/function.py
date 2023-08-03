from numpy import *
import numpy as np
import os
import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import scipy.sparse as sp

def create_resultlist(result,testset,Index_PositiveRow,Index_PositiveCol,Index_zeroRow,Index_zeroCol,test_length_p,zero_length,test_f):
    #result_list = zeros((test_length+zero_length, 1))
    # 创建一个（正样本数量+负样本数量，1）的二维数组
    # 一个test_length_p+len(test_f)行，1列的全零矩阵
    result_list = zeros((test_length_p+len(test_f), 1))
    # 遍历正样本，将模型预测值存入结果列表
    for i in range(test_length_p):
        # result矩阵中的元素按照索引从testset中对应的行和列位置取出并存入结果列表result_list
        result_list[i,0] = result[Index_PositiveRow[testset[i]], Index_PositiveCol[testset[i]]]
    # 遍历负样本，将模型预测值存入结果列表
    for i in range(len(test_f)):
        result_list[i+test_length_p, 0] = result[Index_zeroRow[test_f[i]], Index_zeroCol[test_f[i]]]
    # for i in range(zero_length):
    #     result_list[i+test_length, 0] = result[Index_zeroRow[i],Index_zeroCol[i]]
    return result_list


def create_resultmatrix(result,testset,prolist):
    leave_col = prolist[testset]
    result = result[:,leave_col]
    return result

def create_resultmatrix_row(result,testset,prolist):
    leave_row = prolist[testset]
    result = result[leave_row, :]
    return result

def laplacian_loss(sim,x):
    # sim = torch.from_numpy(sim)
    sim = sim.double()
    x = x.double()
    D = torch.diag(torch.sum(sim, dim=1))  # 对角线为每个节点的度
    L = D - sim
    l_loss = torch.trace(torch.mm(torch.mm(x.t(), L), x))
    return l_loss

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

    # laplacian_matrix = torch.from_numpy(laplacian_matrix)

    return laplacian_matrix

def laplacian_loss(x):
    # sim = torch.from_numpy(sim)
    x = x.double()
    D = torch.diag(torch.sum(x, dim=1))  # 对角线为每个节点的度
    L = D - x
    l_loss = torch.trace(torch.mm(torch.mm(x.t(), L), x)).cuda()
    return 2*l_loss


# def get_edge_index(x):
#     if isinstance(x, torch.Tensor):
#         # Convert the tensor to a COO format
#         row, col = x.nonzero(as_tuple=True)
#         edge_index = torch.stack([row, col], dim=0)
#         # Create a Data object with the edge index
#         edge_index = Data(edge_index, None)
#     else:
#         x_sparse = from_scipy_sparse_matrix(x)
#         edge_index = dense_to_sparse(x_sparse)[0]
#     return edge_index

