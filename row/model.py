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
    def __init__(self, num_in_node = 435, num_in_edge = 757, num_hidden1 = 512, num_out=128):
        super(Model, self).__init__()
        self.node_encoders1 = node_encoder(num_in_edge, num_hidden1, 0.3)

        self.hyperedge_encoders1 = hyperedge_encoder(num_in_node, num_hidden1, 0.3)

        self.decoder1 = decoder1(act=lambda x: x)
        self.decoder2 = decoder2(act=lambda x: x)

        self._enc_mu_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)

        self._enc_mu_hedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_hyedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self.hgnn_node2 = HGNN1(num_in_node, num_in_node, num_out, num_in_node, num_in_node)
        self.hgnn_hyperedge2 = HGNN1(num_in_edge, num_in_edge, num_out, num_in_edge, num_in_edge)

        self.dropout = nn.Dropout(0.8)
        self.act = torch.sigmoid





    def sample_latent(self, z_node, z_hyperedge):
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


        if self.training:
            return self.z_node_, self.z_hyperedge_  # Reparameterization trick
        else:
            return self.z_node_mean, self.z_edge_mean


    def forward(self, AT, A , HMG, HDG, mir_feat, dis_feat, HMD, HDM , HMM , HDD):

        z_node_encoder = self.node_encoders1(AT)
        z_hyperedge_encoder = self.hyperedge_encoders1(A)
        self.z_node_s, self.z_hyperedge_s = self.sample_latent(z_node_encoder, z_hyperedge_encoder)

        mir_feature_1 = self.hgnn_hyperedge2(mir_feat, HMG)
        dis_feature_1 = self.hgnn_node2(dis_feat, HDG)

        mir_feature_2 = self.hgnn_hyperedge2(mir_feat, HMD)
        dis_feature_2 = self.hgnn_node2(dis_feat, HDM)

        mir_feature_3 = self.hgnn_hyperedge2(mir_feat, HMM)
        dis_feature_3 = self.hgnn_node2(dis_feat, HDD)

        reconstructionMMDD = self.decoder1(dis_feature_3, mir_feature_3)
        reconstructionMD = self.decoder1(dis_feature_2, mir_feature_2)
        reconstructionG = self.decoder1(dis_feature_1, mir_feature_1)
        reconstruction_en = self.decoder2(self.z_node_mean, self.z_edge_mean)
        result = self.z_node_mean.mm(self.z_edge_mean.t())
        result_h = (reconstructionG + reconstructionMD + reconstructionMMDD)/3

        recover = 0.1*result + 0.9*result_h

        return reconstruction_en, result, reconstructionG, reconstructionMD, reconstructionMMDD, result_h,recover,mir_feature_1 , mir_feature_2 , mir_feature_3 , dis_feature_1 ,dis_feature_2 ,dis_feature_3


