import numpy as np
import copy
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from model import Model
from numpy.core import multiarray
from hypergraph_utils import *
import os
from kl_loss import kl_loss
from function import create_resultlist
from utils import f1_score_binary,precision_binary,recall_binary
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def contrastive_loss(h1, h2, tau = 0.7):
    sim_matrix = sim(h1, h2)
    f = lambda x: torch.exp(x / tau)
    matrix_t = f(sim_matrix)
    numerator = matrix_t.diag()
    denominator = torch.sum(matrix_t, dim=-1)
    loss = -torch.log(numerator / denominator).mean()
    return loss



def train(epochs):
    auc1 = 0
    aupr1 = 0
    recall1 = 0
    precision1 = 0
    f11 = 0
    if epoch != epochs - 1:
        model.train()
        reconstruction1, result, reconstructionG, reconstructionMD, reconstructionMMDD, result_h ,recover,mir_feature_1 , mir_feature_2 , mir_feature_3 , dis_feature_1 ,dis_feature_2 ,dis_feature_3=  model( AT, A , HMG, HDG, mir_feat, dis_feat,HMD , HDM , HMM , HDD)#将数据传入模型
        outputs = recover .t().cpu().detach().numpy()
        test_predict = create_resultlist(outputs, testset, Index_PositiveRow, Index_PositiveCol, Index_zeroRow,Index_zeroCol, len(test_p), zero_length, test_f)


        MA = torch.masked_select(A, train_mask_tensor)
        reG = torch.masked_select(reconstructionG.t(),train_mask_tensor)
        reMD = torch.masked_select(reconstructionMD.t(), train_mask_tensor)
        reMMDD = torch.masked_select(reconstructionMMDD.t(), train_mask_tensor)
        ret = torch.masked_select(result.t(), train_mask_tensor)
        re1 = torch.masked_select(reconstruction1.t(), train_mask_tensor)
        rec = torch.masked_select(recover.t(), train_mask_tensor)
        loss_k = loss_kl(model.z_node_log_std, model.z_node_mean, model.z_edge_log_std, model.z_edge_mean)

        loss_c_m =  contrastive_loss(mir_feature_2,mir_feature_1) + contrastive_loss(mir_feature_2,mir_feature_3) 
        loss_c_d = contrastive_loss(dis_feature_2, dis_feature_1) + contrastive_loss(dis_feature_2, dis_feature_3) 
        loss_c = loss_c_m + loss_c_d

        loss_v = loss_k + F.binary_cross_entropy_with_logits(re1.t(), MA,pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(ret.t(), MA,pos_weight=pos_weight)
        loss_r_h = F.binary_cross_entropy_with_logits(reG.t(), MA,pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(reMMDD.t(), MA,pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(reMD.t(), MA,pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(rec.t(), MA,pos_weight=pos_weight)

        loss = loss_r_h + alpha*loss_v + (1-alpha)*loss_c

        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        auc_val = roc_auc_score(label, test_predict)
        aupr_val = average_precision_score(label, test_predict)

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.5f}'.format(loss.data.item()),
              'auc_val: {:.5f}'.format(auc_val),
              'aupr_val: {:.5f}'.format(aupr_val),
              )
        max_f1_score, threshold = f1_score_binary(torch.from_numpy(label).float(),torch.from_numpy(test_predict).float())
        print("//////////max_f1_score",max_f1_score)
        precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////precision:", precision)
        recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////recall:", recall)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        
        
    if epoch == args.epochs - 1:
        auc1 = auc_val
        aupr1 = aupr_val
        recall1 = recall
        precision1 = precision
        f11 = max_f1_score

        print('auc_test: {:.5f}'.format(auc1),
              'aupr_test: {:.5f}'.format(aupr1),
              'precision_test: {:.5f}'.format(precision1),
              'recall_test: {:.5f}'.format(recall1),
              'f1_test: {:.5f}'.format(f11),
              )

    return auc1,aupr1,recall1,precision1,f11





MD = np.loadtxt("data/md_delete.txt")
MM = np.loadtxt("data/mm_delete.txt")
DD = np.loadtxt("data/dd_delete.txt")
DG = np.loadtxt("data/dg_delete.txt")
MG = np.loadtxt("data/mg_delete.txt")

[row, col] = np.shape(MD)

indexn = np.argwhere(MD == 0)
Index_zeroRow = indexn[:, 0]
Index_zeroCol = indexn[:, 1]

indexp = np.argwhere(MD == 1)
Index_PositiveRow = indexp[:, 0]
Index_PositiveCol = indexp[:, 1]

totalassociation = np.size(Index_PositiveRow) #7694
fold = int(totalassociation / 5) #1538

zero_length = np.size(Index_zeroRow)#321601

alpha = 0.7
n = 1
hidden1 = 512
hidden2 = 128
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=135, help='Number of epochs to train.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cv_num', type=int, default=5, help='number of fold')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


AAuc_list1 = []
f1_score_list1 = []
precision_list1 = []
recall_list1 = []
aupr_list1 = []

auc_sum = 0
aupr_sum = 0
AUC = 0
AUPR = 0
recall_sum = 0
precision_sum = 0
f1_sum = 0

for time in range(1,n+1):
    Auc_per = []
    f1_score_per = []
    precision_per = []
    recall_per = []
    aupr_per = []
    p = np.random.permutation(totalassociation)
    # print(p)

    auc = 0
    aupr = 0
    rec = 0
    pre = 0
    f1 = 0

    for f in range(1, args.cv_num + 1):
        print("cross_validation:", '%01d' % (f))

        if f == args.cv_num:
            testset = p[((f - 1) * fold): totalassociation + 1]
        else:
            testset = p[((f - 1) * fold): f * fold]

        all_f = np.random.permutation(np.size(Index_zeroRow))

        test_p = list(testset)

        test_f = all_f[0:len(test_p)]

        difference_set_f = list(set(all_f).difference(set(test_f)))
        train_f = difference_set_f

        train_p = list(set(p).difference(set(testset)))

        X = copy.deepcopy(MD)
        Xn = copy.deepcopy(X)

        zero_index = []
        for ii in range(len(train_f)):
            zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])

        true_list = multiarray.zeros((len(test_p) + len(test_f), 1))
        for ii in range(len(test_p)):
            Xn[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
            true_list[ii, 0] = 1
        train_mask = np.ones(shape=Xn.shape)
        for ii in range(len(test_p)):
            train_mask[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
            train_mask[Index_zeroRow[test_f[ii]], Index_zeroCol[test_f[ii]]] = 0
        train_mask_tensor = torch.from_numpy(train_mask).to(torch.bool)


        label = true_list
        HHMG = construct_H_with_KNN(MG)
        HMG = generate_G_from_H(HHMG)
        HMG = HMG.double()

        HHDG = construct_H_with_KNN(DG)
        HDG = generate_G_from_H(HHDG)
        HDG = HDG.double()
        mir_feat = torch.eye(757)
        dis_feat = torch.eye(435)
        parameters = [435, 757]

        A = copy.deepcopy(Xn)
        AT = A.T


        HHMD = construct_H_with_KNN(A)
        HMD = generate_G_from_H(HHMD) #757*757
        HMD = HMD.double()

        HHDM = construct_H_with_KNN(AT)
        HDM = generate_G_from_H(HHDM) #435*435
        HDM = HDM.double()

        HHMM = construct_H_with_KNN(MM)
        HMM = generate_G_from_H(HHMM)
        HMM = HMM.double()

        HHDD = construct_H_with_KNN(DD)
        HDD = generate_G_from_H(HHDD)
        HDD = HDD.double()

        model = Model()
        optimizer2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        A = torch.from_numpy(A)
        AT = torch.from_numpy(AT)
        XX = copy.deepcopy(Xn)
        XX = torch.from_numpy(XX)
        XXN = A
        pos_weight = float(XXN.shape[0] * XXN.shape[1] - XXN.sum()) / XXN.sum()
        # norm = A.shape[0] * A.shape[1] / float((A.shape[0] * A.shape[1] - A.sum()) * 2)
        mir_feat, dis_feat = Variable(mir_feat), Variable(dis_feat)
        loss_kl = kl_loss(435, 757)
        if args.cuda:
            model.cuda()

            XX = XX.cuda()

            A = A.cuda()
            AT = AT.cuda()

            HMG = HMG.cuda()
            HDG = HDG.cuda()

            HMD = HMD.cuda()
            HDM = HDM.cuda()

            HMM = HMM.cuda()
            HDD = HDD.cuda()

            mir_feat = mir_feat.cuda()
            dis_feat = dis_feat.cuda()

            train_mask_tensor = train_mask_tensor.cuda()

        for epoch in range(args.epochs):

            auc1,aupr1,recall1,precision1,f11 = train(epoch)
            auc = auc + auc1
            aupr = aupr + aupr1
            rec = rec + recall1
            pre = pre + precision1
            f1 = f1 + f11

        if f == args.cv_num:
            print('auc: {:.5f}'.format(auc/args.cv_num),
                  'aupr: {:.5f}'.format(aupr/args.cv_num),
                  'precision: {:.5f}'.format(pre / args.cv_num),
                  'recall: {:.5f}'.format(rec / args.cv_num),
                  'f1_score: {:.5f}'.format(f1 / args.cv_num),
                      )
            a = auc/args.cv_num
            b = aupr/args.cv_num
            c = pre / args.cv_num
            d = rec / args.cv_num
            e = f1 / args.cv_num

    auc_sum = auc_sum + a
    aupr_sum = aupr_sum + b
    precision_sum= precision_sum +c
    recall_sum = recall_sum + d
    f1_sum = f1_sum + e

print(
      'auc_ave: {:.5f}'.format(auc_sum/n),
      'aupr_ave: {:.5f}'.format(aupr_sum/n),
      'precision_ave: {:.5f}'.format(precision_sum / n),
      'recall_ave: {:.5f}'.format(recall_sum / n),
      'f1_ave: {:.5f}'.format(f1_sum / n),
                      )  

















