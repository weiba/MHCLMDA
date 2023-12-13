from numpy import *
import numpy as np
import os
import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data
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


