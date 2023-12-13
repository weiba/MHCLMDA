import numpy as np
import csv
import torch as t
import random
from numpy import *
import torch

def gaussian_normalization(x: t.Tensor, dim: None or 0 or 1):
    """
    Gaussian normalization.
    :param x: matrix
    :param dim: dimension, global normalization if None, column normalization if 0, else row normalization
    :return: After normalized matrix
    """
    if dim is None:
        mean = t.mean(x)
        std = t.std(x)
        x = t.div(t.sub(x, mean), std)
    else:
        mean = t.mean(x, dim=dim)
        std = t.std(x, dim=dim)
        if dim:
            x = t.div(t.sub(x, mean.view([-1, 1])), std.view([-1, 1]))
        else:
            x = t.div(t.sub(x, mean.view([1, -1])), std.view([1, -1]))
    return x

def create_resultlist(result,testset,Index_PositiveRow,Index_PositiveCol,Index_zeroRow,Index_zeroCol,test_length_p,zero_length,test_f):
    #result_list = zeros((test_length+zero_length, 1))
    result_list = zeros((test_length_p+len(test_f), 1))
    for i in range(test_length_p):
        result_list[i,0] = result[Index_PositiveRow[testset[i]], Index_PositiveCol[testset[i]]]
    for i in range(len(test_f)):
        result_list[i+test_length_p, 0] = result[Index_zeroRow[test_f[i]], Index_zeroCol[test_f[i]]]
    # for i in range(zero_length):
    #     result_list[i+test_length, 0] = result[Index_zeroRow[i],Index_zeroCol[i]]
    return result_list

def create_resultmatrix(result,testset,prolist):
    leave_col = prolist[testset]
    result = result[:,leave_col]
    return result

def computer_GIP(intMat,gamadd = 1,gamall = 1):
    num_miRNAs, num_diseasesss = intMat.shape
    sd = []
    sl = []

    for i in range(num_miRNAs):
        sd.append((np.linalg.norm(intMat[i, :])) ** 2)
    gamad = num_miRNAs / sum(sd) * gamadd
    for j in range(num_diseasesss):
        sl.append((np.linalg.norm(intMat[:, j])) ** 2)
    gamal = num_diseasesss / sum(sl) * gamall

    kd = np.ones([num_miRNAs, num_miRNAs],dtype=np.float64)
    for i in range(num_miRNAs):
        for j in range(i+1,num_miRNAs):
            kd[i, j] = np.e ** (-gamad * pow(np.linalg.norm(intMat[i, :] - intMat[j, :]),2))
            kd[j, i] = kd[i, j]

    kt = np.ones([num_diseasesss, num_diseasesss],dtype=np.float64)
    for i in range(num_diseasesss):
        for j in range(i+1,num_diseasesss):
            kt[i, j] = np.e ** (-gamal * pow(np.linalg.norm(intMat[:, i] - intMat[:, j]),2))
            kt[j, i] = kt[i, j]
    return kd,kt

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)

# dis  m   g
def get_edge_index(matrix,i_offset,j_offset):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i+i_offset)
                edge_index[1].append(j+j_offset)
    return t.LongTensor(edge_index)

def f1_score_binary(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: true data,torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: max F1 score and threshold
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    with torch.no_grad():
        thresholds = torch.unique(predict_data)
    size = torch.tensor([thresholds.size()[0], true_data.size()[0]], dtype=torch.int32, device=true_data.device)
    ones = torch.ones([size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros([size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.view([1, -1]).ge(thresholds.view([-1, 1])), ones, zeros)
    tpn = torch.sum(torch.where(predict_value.eq(true_data.view([1, -1])), ones, zeros), dim=1)
    tp = torch.sum(torch.mul(predict_value, true_data.view([1, -1])), dim=1)
    two = torch.tensor(2, dtype=torch.float32, device=true_data.device)
    n = torch.tensor(size[1].item(), dtype=torch.float32, device=true_data.device)
    scores = torch.div(torch.mul(two, tp), torch.add(n, torch.sub(torch.mul(two, tp), tpn)))
    max_f1_score = torch.max(scores)
    threshold = thresholds[torch.argmax(scores)]
    return max_f1_score, threshold


def accuracy_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: acc
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    n = true_data.size()[0]
    ones = torch.ones(n, dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(n, dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tpn = torch.sum(torch.where(predict_value.eq(true_data), ones, zeros))
    score = torch.div(tpn, n)
    return score


def precision_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tp = torch.sum(torch.mul(true_data, predict_value))
    true_neg = torch.sub(ones, true_data)
    tf = torch.sum(torch.mul(true_neg, predict_value))
    score = torch.div(tp, torch.add(tp, tf))
    return score


def recall_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tp = torch.sum(torch.mul(true_data, predict_value))
    predict_neg = torch.sub(ones, predict_value)
    fn = torch.sum(torch.mul(predict_neg, true_data))
    score = torch.div(tp, torch.add(tp, fn))
    return score


def mcc_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    predict_neg = torch.sub(ones, predict_value)
    true_neg = torch.sub(ones, true_data)
    tp = torch.sum(torch.mul(true_data, predict_value))
    tn = torch.sum(torch.mul(true_neg, predict_neg))
    fp = torch.sum(torch.mul(true_neg, predict_value))
    fn = torch.sum(torch.mul(true_data, predict_neg))
    delta = torch.tensor(0.00001, dtype=torch.float32, device=true_data.device)
    score = torch.div((tp*tn-fp*fn), torch.add(delta, torch.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
    return score
