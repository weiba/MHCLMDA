import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class hyperedge_encoder(nn.Module):
    def __init__(self, num_in_edge, num_hidden, dropout, act=torch.tanh):
        super(hyperedge_encoder, self).__init__()
        self.num_in_edge = num_in_edge
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act

        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_edge, self.num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

    def forward(self, H_T):
        # print('hyperedge_encoder self.W1.shape',self.W1.shape)
        # print('hyperedge_encoder self.b1.shape', self.b1.shape)
        z1 = self.act(torch.mm(H_T, self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_edge) + ' -> ' + str(self.num_hidden)


class node_encoder(nn.Module):
    def __init__(self, num_in_node, num_hidden, dropout, act=torch.tanh):
        super(node_encoder, self).__init__()
        self.num_in_node = num_in_node
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_node, self.num_hidden), dtype=torch.double))
        # gain=1 auc_ave: 0.93899 aupr_ave: 0.93845
        # gain=0.5 auc_ave: 0.93807 aupr_ave: 0.93816
        # gain = 1.414 auc_ave: 0.93915 aupr_ave: 0.93894  auc_ave: 0.93848 aupr_ave: 0.93797
        # nn.init.xavier_uniform_(self.W1.data, gain=1)
        # nn.init.xavier_uniform_(self.W1.data, gain=0.5)
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

    def forward(self, H):
        # print('node_encoder self.W1.shape',self.W1.shape)
        # print('node_encoder self.b1.shape', self.b1.shape)
        z1 = self.act(H.mm(self.W1) + 2*self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_node) + ' -> ' + str(self.num_hidden)


class decoder2(nn.Module):
    # 0.9 auc_ave: 0.93781 aupr_ave: 0.93706
    # 0.85 auc_ave: 0.93783 aupr_ave: 0.93701
    # 0.8 auc_ave: 0.93915 aupr_ave: 0.93894  auc_ave: 0.93848 aupr_ave: 0.93797
    # 0.7 auc_ave: 0.93713 aupr_ave: 0.93706
    #
    def __init__(self, dropout=0.8, act=torch.sigmoid):
        super(decoder2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z_node, z_hyperedge):
        z_node_ = self.dropout(z_node)
        z_hyperedge_ = self.dropout(z_hyperedge)

        # z_node_ = z_node #435, 435 权重后的是435*128
        # z_hyperedge_ = z_hyperedge #757, 757  权重后的是757*128

        z = self.act(z_node_.mm(z_hyperedge_.t()))
        return z

class decoder1(nn.Module):
    def __init__(self, dropout=0.5, act=torch.sigmoid):
        super(decoder1, self).__init__()

        # 在这个代码中，nn.Dropout(0.5)表示在网络的训练过程中，有50%的神经元会被随机“不工作”，也就是说，这50%的神经元的权重在计算损失函数时不会被用到。
        # 在测试过程中，所有的神经元都会参与计算，这样能使网络更稳定。
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z_node, z_hyperedge):
        # z_node_ = self.dropout(z_node) #435, 435 权重后的是435*128
        # z_hyperedge_ = self.dropout(z_hyperedge) #757, 757  权重后的是757*128

        z_node_ = z_node #435, 435 权重后的是435*128
        z_hyperedge_ = z_hyperedge #757, 757  权重后的是757*128

        # 其中 mm 是Pytorch中的矩阵乘法，它的作用是计算矩阵 z_node_ 与矩阵 z_hyperedge_ 的转置的乘积，将得到的结果作为变量 z 的值。
        z = self.act(z_node_.mm(z_hyperedge_.t()))

        return z

class HGNN_conv1(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):  #
        super(HGNN_conv1, self).__init__()

        self.weight = Parameter(torch.DoubleTensor(in_ft, out_ft)) #权重参数
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_ft)) #偏置
        else:
            self.register_parameter('bias', None)
        #初始化权重和偏置参数
        self.linear_x_1 = nn.Linear(in_ft, out_ft).to(torch.double)
        # self.batch = nn.BatchNorm1d(out_ft).to(torch.double)
        self.reset_parameters()




    def reset_parameters(self):
        # 根据输入和输出维度计算出初始标准差stdv
        stdv = 1. / math.sqrt(self.weight.size(1))
        # 使用均匀分布初始化权重参数，范围为[-stdv, stdv]
        self.weight.data.uniform_(-stdv, stdv)
        # 如果有偏置项，则使用均匀分布初始化偏置参数，范围同样为[-stdv, stdv]
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):  # x: torch.Tensor, G: torch.Tensor

        # part1
        x = x.double()
        x = x.matmul(self.weight) #757*512  757*128
        # x1 = self.batch(x1)
        # x1 = self.linear_x_1(x1) # 757*512  757*128
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x) + x #757*512  757*128


        return x

class HGNN1(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, n_node, emb_dim, n_hid_2=128,dropout=0.5):
        super(HGNN1, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv1(in_ch, n_hid)
        self.hgc2 = HGNN_conv1(n_hid, n_class)
        # self.hgc2 = HGNN_conv1(n_hid, n_hid_2)
        # self.hgc3 = HGNN_conv1(n_hid_2, n_class)

        # self.feat = nn.Embedding(n_node, emb_dim)
        # self.feat_idx = torch.arange(n_node).cuda()
        # nn.init.xavier_uniform_(self.feat.weight)
        # self.weight1 = Parameter(torch.Tensor(in_ch, n_hid))
        # self.weight2 = Parameter(torch.Tensor(n_hid, n_class))

    def forward(self, x, G):
        G = G + torch.eye(G.shape[0]).cuda()
        x= self.hgc1(x, G) #x1,x2,x3,757*512
        x = torch.tanh(x)
        x= self.hgc2(x, G)


        return x

class HGNN2(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, n_node, emb_dim, n_hid_2=128,dropout=0.5):
        super(HGNN2, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv1(in_ch, n_hid)

        self.hgc2 = HGNN_conv1(n_hid, n_hid_2)
        # self.hgc3 = HGNN_conv1(n_hid_2, n_class)

        # self.feat = nn.Embedding(n_node, emb_dim)
        # self.feat_idx = torch.arange(n_node).cuda()
        # nn.init.xavier_uniform_(self.feat.weight)
        # self.weight1 = Parameter(torch.Tensor(in_ch, n_hid))
        # self.weight2 = Parameter(torch.Tensor(n_hid, n_class))

    def forward(self, x, G):
        G = G + torch.eye(G.shape[0]).cuda()
        x= self.hgc1(x, G) #x1,x2,x3,757*512
        x = torch.tanh(x)


        return x





