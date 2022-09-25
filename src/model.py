import pyro

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, dropout=0.0, bias=True):
        super(GCNLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = dropout

    def forward(self, adj, x):
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.linear(x)
        h = adj @ h

        if self.activation:
            h = self.activation(h)

        return h


# GNN Classifier for implementing p(Y|A,X)
class GCN_Classifier(nn.Module):
    
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN_Classifier, self).__init__()

        self.shared_gcn1 = GCNLayer(in_dim, hid_dim)
        self.shared_gcn2 = GCNLayer(hid_dim, hid_dim)

        self.t_head = nn.Linear(hid_dim, out_dim)
        self.s_head = nn.Linear(hid_dim, out_dim)

        self.t_gcn1 = GCNLayer(in_dim, hid_dim)
        self.t_gcn2 = GCNLayer(hid_dim, out_dim)

        self.s_linear1 = nn.Linear(in_dim, hid_dim)
        self.s_linear2 = nn.Linear(hid_dim, out_dim)
        
        self.dropout = dropout

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.shared_gcn1(adj, x))

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.shared_gcn2(adj, x))

        return x

    def teacher_head(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.t_head(x)

        return x

    def student_head(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.s_head(x) 

        return x

    def teacher_net(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.t_gcn1(adj, x))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.t_gcn2(adj, x)

        return x

    def student_net(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.s_linear1(x))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.s_linear2(x)      

        return x


# Graph Augmentation Module - GraphAug for implementing p(\widehat{A}|A,X)
class Augmentor(nn.Module):
    def __init__(self, in_dim, hid_dim, alpha, temp, dataset):
        super(Augmentor, self).__init__()

        self.gcn1 = GCNLayer(in_dim, hid_dim, None, 0, bias=False)
        self.gcn2 = GCNLayer(hid_dim, hid_dim, F.relu, 0, bias=False)

        self.alpha = alpha
        self.temp = temp
        self.dataset = dataset

    def forward(self, x, adj_norm, adj_orig):

        # Parameterized Augmentation Distribution
        h = self.gcn1(adj_norm, x)
        h = self.gcn2(adj_norm, h)
        adj_logits = h @ h.T

        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = self.alpha * edge_probs + (1-self.alpha) * adj_orig

        # Gumbel-Softmax Sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temp, probs=edge_probs).rsample()
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T

        adj_sampled.fill_diagonal_(1)
        D_norm = torch.diag(torch.pow(adj_sampled.sum(1), -0.5))
        adj_sampled = D_norm @ adj_sampled @ D_norm

        return adj_sampled, adj_logits


# KL-divergence Loss for knowledge distillation
def com_distillation_loss(t_logits, s_logits, adj_orig, adj_sampled, temp, loss_mode):

    s_dist = F.log_softmax(s_logits / temp, dim=-1)
    t_dist = F.softmax(t_logits / temp, dim=-1)
    if loss_mode == 0:
        kd_loss = temp * temp * F.kl_div(s_dist, t_dist)
    elif loss_mode == 1:
        kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())

    adj = torch.triu(adj_orig * adj_sampled).detach()
    edge_list = (adj + adj.T).nonzero().t()

    s_dist_neigh = F.log_softmax(s_logits[edge_list[0]] / temp, dim=-1)
    t_dist_neigh = F.softmax(t_logits[edge_list[1]] / temp, dim=-1)
    if loss_mode == 0:
        kd_loss += temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh)
    elif loss_mode == 1:
        kd_loss += temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh.detach())

    return kd_loss