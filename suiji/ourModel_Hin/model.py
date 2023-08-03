from layer import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module


class Model(nn.Module):
    def __init__(self, num_in_node = 435, num_in_edge = 757, num_hidden1 = 512, num_out=128):  # 435, 757, 512, 128
        super(Model, self).__init__()
        self.node_encoders1 = node_encoder(num_in_edge, num_hidden1, 0.3)

        self.hyperedge_encoders1 = hyperedge_encoder(num_in_node, num_hidden1, 0.3)

        self.decoder1 = decoder1(act=lambda x: x)
        self.decoder2 = decoder2(act=lambda x: x)

        self._enc_mu_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)

        self._enc_mu_hedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_hyedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)

        # self.hgnn_node2 = HGNN1(num_in_node, num_in_node, num_out, num_in_node, num_in_node)  # 435,512,128,435,435
        # self.hgnn_hyperedge2 = HGNN1(num_in_edge, num_in_edge, num_out, num_in_edge, num_in_edge)  # 757,512,128,757,757

        self.hgnn_node2 = HGNN1(num_in_node, num_in_node, num_out, num_in_node, num_in_node)  # 435,512,128,435,435
        self.hgnn_hyperedge2 = HGNN1(num_in_edge, num_in_edge, num_out, num_in_edge, num_in_edge)  # 757,512,128,757,757

        self.dropout = nn.Dropout(0.8)
        self.act = torch.sigmoid

        # self.batch = nn.BatchNorm1d(128).double()

        self.weight = Parameter(torch.DoubleTensor(num_out, num_out))  # 权重参数
        nn.init.xavier_uniform_(self.weight)



    def sample_latent(self, z_node, z_hyperedge):
        # Return the latent normal sample z ~ N(mu, sigma^2)
        self.z_node_mean = self._enc_mu_node(z_node)  # mu
        self.z_node_log_std = self._enc_log_sigma_node(z_node)
        self.z_node_std = torch.exp(self.z_node_log_std)  # sigma
        z_node_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_node_std.size())).double()
        self.z_node_std_ = z_node_std_.cuda()
        self.z_node_ = self.z_node_mean + self.z_node_std.mul(Variable(self.z_node_std_, requires_grad=True))

        self.z_edge_mean = self._enc_mu_hedge(z_hyperedge)
        self.z_edge_log_std = self._enc_log_sigma_hyedge(z_hyperedge)
        self.z_edge_std = torch.exp(self.z_edge_log_std)  # sigma
        z_edge_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_edge_std.size())).double()
        self.z_edge_std_ = z_edge_std_.cuda()
        self.z_hyperedge_ = self.z_edge_mean + self.z_edge_std.mul(Variable(self.z_edge_std_, requires_grad=True))



        # 重参数化 Reparameterization Trick 如果处于训练状态，就会对节点和超边分别进行采样操作 如果处于测试状态，就返回节点和超边的均值
        if self.training:
            return self.z_node_, self.z_hyperedge_  # Reparameterization trick
        else:
            return self.z_node_mean, self.z_edge_mean


    def forward(self, AT, A , HMG, HDG, mir_feat, dis_feat, HMD, HDM , HMM , HDD):
        # side embedding
        z_node_encoder = self.node_encoders1(AT) #435*512
        z_hyperedge_encoder = self.hyperedge_encoders1(A) #757*512

        # self.z_node_s torch.Size([435, 128])
        # self.z_hyperedge_s torch.Size([757, 128])
        self.z_node_s, self.z_hyperedge_s = self.sample_latent(z_node_encoder, z_hyperedge_encoder)
        z_node = self.z_node_s #435*128
        # print('z_node.shape',z_node.shape)
        z_hyperedge = self.z_hyperedge_s #757*128
        # print('z_hyperedge.shape',z_hyperedge.shape)

        # dis_feat = self.act(self.z_node_s.mm(self.z_node_s.t()))
        # mir_feat= self.act(self.z_hyperedge_s.mm(self.z_hyperedge_s.t()))

        mir_feature_1 = self.hgnn_hyperedge2(mir_feat, HMG) # 757*128
        dis_feature_1 = self.hgnn_node2(dis_feat, HDG) # 435*128

        mir_feature_2 = self.hgnn_hyperedge2(mir_feat, HMD) # 757*128
        dis_feature_2 = self.hgnn_node2(dis_feat, HDM) # 435*128


        # mir_feature_2_1 = F.normalize(mir_feature_2)
        # dis_feature_2_1 = F.normalize(dis_feature_2)
        #
        # mir_feature_2_1 = mir_feature_2_1.mm(self.weight)
        # dis_feature_2_1 = dis_feature_2_1.mm(self.weight)
        #
        # lm = mir_feature_2_1.mm(mir_feature_2_1.t())
        # ld = dis_feature_2_1.mm(dis_feature_2_1.t())

        mir_feature_3 = self.hgnn_hyperedge2(mir_feat, HMM) # 757*128
        dis_feature_3 = self.hgnn_node2(dis_feat, HDD) # 435*128

        reconstructionMMDD = self.decoder1(dis_feature_3, mir_feature_3)
        reconstructionMD = self.decoder1(dis_feature_2, mir_feature_2)
        reconstructionG = self.decoder1(dis_feature_1, mir_feature_1)



        # 换为z_node_mean后 auc_ave: 0.94071 aupr_ave: 0.94108
        # reconstruction_en = self.decoder2(self.z_node_mean, self.z_edge_mean)
        reconstruction_en = self.decoder2(z_node, z_hyperedge)
        result = self.z_node_mean.mm(self.z_edge_mean.t())
        # result = self.act(self.dropout(self.z_node_mean).mm(self.dropout(self.z_edge_mean).t()))
        # result_h = (reconstructionG + reconstructionMD + reconstructionMMDD)/3
        result_h = 0.2*reconstructionG + 0.6*reconstructionMD + 0.2*reconstructionMMDD
        # recover = (result + result_h)/2
        # recover = 0.8*result + 0.2*result_h auc_ave: 0.93979 aupr_ave: 0.94000
        # recover = 0.7*result + 0.3*result_h auc_ave: 0.94086 aupr_ave: 0.94124
        # recover = 0.9*result + 0.1*result_h auc_ave: 0.93868 aupr_ave: 0.93942
        # recover = 0.55*result + 0.45*result_h auc_ave: 0.94099 aupr_ave: 0.94182
        # recover = 0.65*result + 0.35*result_h auc_ave: 0.94109 aupr_ave: 0.94196
        # recover = 0.6*result + 0.4*result_h auc_ave: 0.94191 aupr_ave: 0.94245
        # recover = 0.4*result + 0.6*result_h auc_ave: 0.94012 aupr_ave: 0.94090
        # recover = 0.3*result + 0.7*result_h auc_ave: 0.93971 aupr_ave: 0.94054
        recover = 0.6*result + 0.4*result_h

        return reconstruction_en, result, reconstructionG, reconstructionMD, reconstructionMMDD, result_h ,recover,mir_feature_1 , mir_feature_2 , mir_feature_3 , dis_feature_1 ,dis_feature_2 ,dis_feature_3


