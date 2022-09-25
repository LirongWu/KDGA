import torch
import pickle
import numpy as np
import scipy.sparse as sp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # Load homophily and heterophily graph datasets
def load_data(dataset):

    adj = pickle.load(open(f'../data/graphs/{dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'../data/graphs/{dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'../data/graphs/{dataset}_labels.pkl', 'rb'))
    data_mask = pickle.load(open(f'../data/graphs/{dataset}_tvt_nids.pkl', 'rb'))

    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())
    else:
        features = torch.FloatTensor(features)
    if dataset == 'cora' or dataset == 'citeseer':
        features = torch.nn.functional.normalize(features, p=1, dim=1)

    adj.setdiag(1)
    adj_orig = scipysp_to_pytorchsp(adj).to_dense()

    degrees = np.array(adj.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
    adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
    adj_norm = scipysp_to_pytorchsp(adj_norm).to_dense()

    labels = torch.LongTensor(labels)
    nclass = len(torch.unique(labels))
    
    train_mask = data_mask[0]
    val_mask = data_mask[1]
    test_mask = data_mask[2]

    return features.to(device), adj_orig.to(device), adj_norm.to(device), labels.to(device), train_mask, val_mask, test_mask, nclass


def scipysp_to_pytorchsp(sp_mx):

    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()

    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape

    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T), torch.FloatTensor(values), torch.Size(shape))

    return pyt_sp_mx