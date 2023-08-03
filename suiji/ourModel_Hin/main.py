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
from function import create_resultlist , laplacian_loss
from utils import f1_score_binary,precision_binary,recall_binary
from sklearn.metrics import precision_recall_curve

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def contrastive_loss(h1, h2, tau = 0.5):
    sim_matrix = sim(h1, h2)
    f = lambda x: torch.exp(x / tau)
    matrix_t = f(sim_matrix)
    numerator = matrix_t.diag()
    denominator = torch.sum(matrix_t, dim=-1)
    loss = -torch.log(numerator / denominator).mean()
    return loss

def cos_dis(x):
    """
    Calculate the cosine distance among each row of x
    :param x: N x D
                N: the object number
                D: Dimension of the feature
    :return: N x N distance matrix
    """
    x = np.mat(x)
    norm = np.linalg.norm(x, axis=1)
    norm[norm == 0] = 1
    x = np.divide(x, norm.reshape(-1, 1))
    dis_mat = 1 - np.matmul(x, x.T)
    return dis_mat

def GIP_kernel (Asso_RNA_Dis):
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    #initate a matrix as result matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    #calculate the result matrix
    for i in range(nc):
        for j in range(nc):
            #calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i,:] - Asso_RNA_Dis[j,:]))
            if r == 0:
                matrix[i][j]=0
            elif i==j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e**(-temp_up/r)
    return matrix
def getGosiR (Asso_RNA_Dis):
# calculate the r in GOsi Kerel
    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i,:])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r

def train(epochs):
    auc1 = 0
    aupr1 = 0
    if epoch != epochs - 1:
        model.train()
        # optimizer2.zero_grad()
        reconstruction1, result, reconstructionG, reconstructionMD, reconstructionMMDD, result_h ,recover,mir_feature_1 , mir_feature_2 , mir_feature_3 , dis_feature_1 ,dis_feature_2 ,dis_feature_3=  model( AT, A , HMG, HDG, mir_feat, dis_feat,HMD , HDM , HMM , HDD)#将数据传入模型
        outputs = recover .t().cpu().detach().numpy()
        test_predict = create_resultlist(outputs, testset, Index_PositiveRow, Index_PositiveCol, Index_zeroRow,Index_zeroCol, len(test_p), zero_length, test_f)
        # result = torch.masked_select(reconstructionMMDD.t(), train_mask_tensor)

        # loss_l = 0.01*laplacian_loss(lm) + 0.01*laplacian_loss(ld)

        MA = torch.masked_select(A, train_mask_tensor)
        reG = torch.masked_select(reconstructionG.t(),train_mask_tensor)
        reMD = torch.masked_select(reconstructionMD.t(), train_mask_tensor)
        reMMDD = torch.masked_select(reconstructionMMDD.t(), train_mask_tensor)
        ret = torch.masked_select(result.t(), train_mask_tensor)
        reh = torch.masked_select(result_h.t(), train_mask_tensor)
        re1 = torch.masked_select(reconstruction1.t(), train_mask_tensor)

        loss_k = loss_kl(model.z_node_log_std, model.z_node_mean, model.z_edge_log_std, model.z_edge_mean)

        loss_c_m =  contrastive_loss(mir_feature_2,mir_feature_1) + contrastive_loss(mir_feature_2,mir_feature_1)
        loss_c_d = contrastive_loss(dis_feature_2, dis_feature_1) + contrastive_loss(dis_feature_2, dis_feature_3)
        loss_c = loss_c_m + loss_c_d

        loss_v = 0.2*loss_k + (1-norm)*F.binary_cross_entropy_with_logits(re1.t(), MA,pos_weight=pos_weight) + (1-norm)*F.binary_cross_entropy_with_logits(ret.t(), MA,pos_weight=pos_weight)
        loss_hg = F.binary_cross_entropy_with_logits(reG.t(), MA,pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(reMD.t(), MA,pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(reMMDD.t(), MA,pos_weight=pos_weight)+ F.binary_cross_entropy_with_logits(reh.t(), MA,pos_weight=pos_weight)
        # 811 auc_ave: 0.94363 aupr_ave: 0.94358
        # 802 auc_ave: 0.94541 aupr_ave: 0.94611
        # 703
        loss = 0.8*loss_v + 0.1*(norm*loss_hg) + 0.1*(norm*loss_c)
        # loss = 0.8*loss_v + norm*loss_hg + 0.02*(norm*loss_c)
        # loss = loss_v + norm*loss_hg + loss_c


        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        auc_val = roc_auc_score(label, test_predict)
        aupr_val = average_precision_score(label, test_predict)
        # print('auc',auc_val)
        # print('apur',aupr_val)
        # print('loss',loss.data.item())
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.5f}'.format(loss.data.item()),
              'auc_val: {:.5f}'.format(auc_val),
              'aupr_val: {:.5f}'.format(aupr_val),
              )
        max_f1_score, threshold = f1_score_binary(torch.from_numpy(label).float(),torch.from_numpy(test_predict).float())
        f1_score_per.append(max_f1_score)
        print("//////////max_f1_score",max_f1_score)
        precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        precision_per.append(precision)
        print("//////////precision:", precision)
        recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        recall_per.append(recall)
        print("//////////recall:", recall)
        # pr, re, thresholds = precision_recall_curve(label, test_predict)
        varf1_score.append(max_f1_score)
        varprecision.append(precision)
        varrecall.append(recall)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


    if epoch == args.epochs - 1:
        auc1 = auc_val
        aupr1 = aupr_val
        print('auc_test: {:.5f}'.format(auc1),
              'aupr_test: {:.5f}'.format(aupr1),
              'precision_test: {:.5f}'.format(precision),
              'recall_test: {:.5f}'.format(recall),
              'f1_test: {:.5f}'.format(max_f1_score),
              )

    return auc1,aupr1





MD = np.loadtxt('data/md_delete.txt')#757*435
MM = np.loadtxt('data/mm_delete.txt')#757*757
DD = np.loadtxt('data/dd_delete.txt')#435*435
DG = np.loadtxt('data/dg_delete.txt')#435*11216
MG = np.loadtxt('data/mg_delete.txt')#757*11216

[row, col] = np.shape(MD)

# 获得负样本的坐标
indexn = np.argwhere(MD == 0)
Index_zeroRow = indexn[:, 0] #负样本的行索引
Index_zeroCol = indexn[:, 1] #负样本的列索引

# 获得正样本的坐标
indexp = np.argwhere(MD == 1)
Index_PositiveRow = indexp[:, 0] #正样本的行索引
Index_PositiveCol = indexp[:, 1] #正样本的列索引

#获取正样本的数量
totalassociation = np.size(Index_PositiveRow) #7694
# print(totalassociation)
fold = int(totalassociation / 5) #1538

#获取负样本的数量
zero_length = np.size(Index_zeroRow)#321601

n = 10
hidden1 = 512
hidden2 = 128
# dropout = 0.5
# epochs = 300

parser = argparse.ArgumentParser()
# epochs 136
parser.add_argument('--epochs', type=int, default=136, help='Number of epochs to train.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
# lr = 0.002 auc_ave: 0.93915 aupr_ave: 0.93894  auc_ave: 0.93848 aupr_ave: 0.93797
#
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')

# 5e-5 auc_ave: 0.93745 aupr_ave: 0.93703
# 5e-4 auc_ave: 0.93781 aupr_ave: 0.93706
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cv_num', type=int, default=5, help='number of fold')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

varauc = []
AAuc_list1 = []

varf1_score = []
f1_score_list1 = []
varprecision = []
precision_list1 = []
varrecall = []
recall_list1 = []
varaupr = []
aupr_list1 = []

auc_sum = 0
aupr_sum = 0
AUC = 0
AUPR = 0

for time in range(1,n+1):
    Auc_per = []
    f1_score_per = []
    precision_per = []
    recall_per = []
    aupr_per = []
    # 一个从0到totalassociation-1的随机数排列p
    p = np.random.permutation(totalassociation)
    # print(p)

    auc = 0
    aupr = 0

    for f in range(1, args.cv_num + 1):
        print("cross_validation:", '%01d' % (f))

        # 对正样本进行分组
        if f == args.cv_num:
            testset = p[((f - 1) * fold): totalassociation + 1]
        else:
            testset = p[((f - 1) * fold): f * fold]

        # 随机生成所有负样本索引的乱序
        all_f = np.random.permutation(np.size(Index_zeroRow))

        #正样本的行索引
        test_p = list(testset)


        # 从所有负样本中随机抽出长度等于正样本数量的样本
        test_f = all_f[0:len(test_p)]

        # 将所有的负样本索引（all_f）转换为集合，然后用这个集合与当前验证集中包含的负样本索引（test_f）的集合的差集作为结果。这里的差集代表的是在all_f中出现而不在test_f中出现的索引。
        difference_set_f = list(set(all_f).difference(set(test_f)))
        train_f = difference_set_f

        # 创建训练数据的正样本的索引列表（train_p）。
        # 它通过将总的正样本索引列表（p）和测试数据的正样本的索引列表（testset）作差来得到。
        # 也就是说，这个train_p列表里面的元素是p列表中没有出现在testset列表中的元素。
        train_p = list(set(p).difference(set(testset)))

        X = copy.deepcopy(MD)
        Xn = copy.deepcopy(X)
        # print('Xn_180', np.sum(Xn))

        zero_index = []
        # 它通过遍历训练负样本的索引，将训练负样本的坐标加入 zero_index 列表
        for ii in range(len(train_f)):
            zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])

        #存储测试样本真实结果（0或1）的矩阵 true_list
        # 将 true_list 初始化为长度为测试正样本数量和测试负样本数量的矩阵，其中每个元素均为0。
        # true_list = zeros((len(test_p) + len(test_f), 1))
        true_list = multiarray.zeros((len(test_p) + len(test_f), 1))  #(3076, 1)
        # print(true_list.shape)


        # 对于当前的测试集中的正样本（即test_p），在二维数组Xn的对应位置将值设为0
        # 在二维数组Xn中，正样本的值被重置为0，而true_list数组用来标记这些位置是正样本
        for ii in range(len(test_p)):
            Xn[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
            true_list[ii, 0] = 1
        # print('Xn_200',np.sum(Xn))
        # print('true_list',len(true_list))

        train_mask = np.ones(shape=Xn.shape)
        for ii in range(len(test_p)):
            train_mask[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
            train_mask[Index_zeroRow[test_f[ii]], Index_zeroCol[test_f[ii]]] = 0
        train_mask_tensor = torch.from_numpy(train_mask).to(torch.bool)


        label = true_list
        # print('label',len(label))3076
        # print(label.shape)


        #超图转换
        CMG = cos_dis(MG)
        HHMG = construct_H_with_KNN(MG) + np.eye(757)
        HMG = generate_G_from_H(HHMG)
        HMG = HMG.double()

        CDG = cos_dis(DG)
        HHDG = construct_H_with_KNN(DG) + np.eye(435)
        HDG = generate_G_from_H(HHDG)
        HDG = HDG.double()
        # print(HDG.shape)
        mir_feat = torch.eye(757) #单位阵
        dis_feat = torch.eye(435)#单位阵
        parameters = [435, 757]

        A = copy.deepcopy(Xn)
        # print('A_232', np.sum(A))
        AT = A.T

        GMD = GIP_kernel(A)
        HHMD = construct_H_with_KNN(A) + np.eye(757)
        HMD = generate_G_from_H(HHMD) #757*757
        HMD = HMD.double()
        # print(HMD.shape)

        GDM = GIP_kernel(AT)
        HHDM = construct_H_with_KNN(AT) + np.eye(435)
        HDM = generate_G_from_H(HHDM) #435*435
        HDM = HDM.double()
        # print(HDM.shape)

        FMM = (MM + GMD) / 2
        HHMM = construct_H_with_KNN(MM+GMD) + np.eye(757)
        HMM = generate_G_from_H(HHMM)
        HMM = HMM.double()

        FDD = (DD + GDM)/2
        HHDD = construct_H_with_KNN(DD+GDM) + np.eye(435)
        HDD = generate_G_from_H(HHDD)
        HDD = HDD.double()
        #
        # TMM = HMM
        # TDD = HDD
        #
        # NMM = generate_G_from_H(MM)
        # NDD = generate_G_from_H(DD)
        # TMM = NMM
        # TDD = NDD

        # TMM = torch.from_numpy(MM)
        # TDD = torch.from_numpy(DD)

        # HM = HMG + HMM + HMD
        # HD = HDG + HDD + HDM

        model = Model()
        optimizer2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        A = torch.from_numpy(A)
        AT = torch.from_numpy(AT)
        XX = copy.deepcopy(Xn)
        XX = torch.from_numpy(XX)
        # XXN = torch.from_numpy(A)
        XXN = A

        # 负样本与正样本的比例
        pos_weight = float(XXN.shape[0] * XXN.shape[1] - XXN.sum()) / XXN.sum()
        norm = A.shape[0] * A.shape[1] / float((A.shape[0] * A.shape[1] - A.sum()) * 2)



        mir_feat, dis_feat = Variable(mir_feat), Variable(dis_feat)
        loss_kl = kl_loss(435, 757)
        if args.cuda:
            model.cuda()

            XX = XX.cuda()

            A = A.cuda()
            AT = AT.cuda()

            HMG = HMG.cuda()
            HDG = HDG.cuda()

            # auc_ave: 0.94400 aupr_ave: 0.94375
            # HMD = HM.cuda()
            # HDM = HD.cuda()
            HMD = HMD.cuda()
            HDM = HDM.cuda()

            HMM = HMM.cuda()
            HDD = HDD.cuda()

            mir_feat = mir_feat.cuda()
            dis_feat = dis_feat.cuda()

            train_mask_tensor = train_mask_tensor.cuda()

            # print('G3.shape:',(G3.shape),
            #       'G4.shape:',(G4.shape),
            #       'TMM.shape:',(TMM.shape),
            #       'TDD.shape:',(TDD.shape),
            #       )





        # auc = 0
        # aupr = 0

        for epoch in range(args.epochs):

            auc1, aupr1 = train(epoch)
            # print('auc1: {:.5f}'.format(auc1),
            #           'aupr1: {:.5f}'.format(aupr1),
            #           )
            auc = auc + auc1
            aupr = aupr + aupr1

        # print('auc_1: {:.5f}'.format(auc),
        #       'aupr_1: {:.5f}'.format(aupr)
        #       )
        # auc_sum = auc_sum + auc1
        # aupr_sum = aupr_sum + aupr1


        if f == args.cv_num:
            print('auc: {:.5f}'.format(auc/args.cv_num),
                  'aupr: {:.5f}'.format(aupr/args.cv_num),
                      )
            a = auc/args.cv_num
            b = aupr/args.cv_num


    auc_sum = auc_sum + a
    aupr_sum = aupr_sum + b
    # print('auc_sum: {:.5f}'.format(auc_sum),
    #       'aupr_sum: {:.5f}'.format(aupr_sum),
    #       )

f1_score_list1.append(np.mean(f1_score_per))
precision_list1.append(np.mean(precision_per))
recall_list1.append(np.mean(recall_per))


print("//////////f1_scoreaverage: " + str(f1_score_list1))
print("//////////precisionaverage: " + str(precision_list1))
print("//////////recallaverage: " + str(recall_list1))


vf1_score = np.var(varf1_score)
vprecision = np.var(varprecision)
vrecall = np.var(varrecall)
print(
      'auc_ave: {:.5f}'.format(auc_sum/n),
      'aupr_ave: {:.5f}'.format(aupr_sum/n),

                      )




















