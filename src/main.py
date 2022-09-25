import os
import csv
import nni
import time
import json
import argparse
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from utils import *
from model import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Pretrain Graph Augmentation Module - GraphAug
def pretrain_Augmentor(model):

    optimizer = torch.optim.Adam(model.parameters(), lr=float(param['lr']))
    
    model.train()
    for epoch in range(param['ga_epochs']):

        _, adj_logits = model(features, adj_norm, adj_orig)
        loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\033[0;30;43m Pretrain EdgePredictor, Epoch [{}/{}]: GA Loss {:.6f}\033[0m'.format(epoch+1, param['ga_epochs'], loss.item()))


# Pretrain GNN Classifier
def pretrain_Classifier(model):

    optimizer = torch.optim.Adam(model.parameters(), lr=float(param['lr']), weight_decay=param['weight_decay'])

    es = 0
    test_best = 0
    test_val = 0
    val_best = 0

    for epoch in range(param['nc_epochs']):

        model.train()

        if param['loss_mode'] == 0:
            t_logits = model.teacher_head(model(adj_norm, features))
            t_loss = nn.CrossEntropyLoss()(t_logits[train_mask], labels[train_mask])
            s_logits = model.student_head(model(adj_norm, features))
            s_loss = nn.CrossEntropyLoss()(s_logits[train_mask], labels[train_mask])
            loss = t_loss + s_loss
        elif param['loss_mode'] == 1:
            t_logits = model.teacher_net(adj_norm, features)
            t_loss = nn.CrossEntropyLoss()(t_logits[train_mask], labels[train_mask])
            s_logits = model.student_net(features)
            s_loss = nn.CrossEntropyLoss()(s_logits[train_mask], labels[train_mask])
            loss = t_loss + t_loss  
        elif param['loss_mode'] == 2:
            logits = model.teacher_net(adj_norm, features)
            loss = nn.CrossEntropyLoss()(logits[train_mask], labels[train_mask])
            s_loss = t_loss = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        if param['loss_mode'] == 0:
            logits_eval = model.teacher_head(model(adj_norm, features))
        elif param['loss_mode'] == 1:
            logits_eval = model.teacher_net(adj_norm, features)
        elif param['loss_mode'] == 2:
            logits_eval = model.teacher_net(adj_norm, features)
        train_acc = evaluation(logits_eval[train_mask], labels[train_mask])
        val_acc = evaluation(logits_eval[val_mask], labels[val_mask])
        test_acc = evaluation(logits_eval[test_mask], labels[test_mask])

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            es = 0
        else:
            es += 1
            if es == 50:
                print('Early stop!')
                break

        print('\033[0;30;41m Pretrain Classifier, Epoch [{}/{}]: Teacher Loss {:.6f}, Student Loss {:.6f} | Train Acc {:.4f}, Val Acc {:.4f}, Test Acc {:.4f} | {:.4f}, {:.4f}\033[0m'.format(
            epoch+1, param['nc_epochs'], t_loss.item(), s_loss.item(), train_acc, val_acc, test_acc, test_val, test_best))


def main():
    
    model_augmentor = Augmentor(features.shape[1], param['hid_dim'], param['alpha'], param['temp_ga'], param['dataset']).to(device)
    model_classifier = GCN_Classifier(features.shape[1], param['hid_dim'], nclass, param['dropout']).to(device)

    if param['ga_epochs']:
        pretrain_Augmentor(model_augmentor)
    if param['nc_epochs']:
        pretrain_Classifier(model_classifier)

    optimizer = torch.optim.Adam(list(model_augmentor.parameters())+list(model_classifier.parameters()), lr=float(param['lr']), weight_decay=param['weight_decay'])

    es = 0
    test_best = 0
    test_val = 0
    val_best = 0

    for epoch in range(param['n_epochs']):
        
        model_augmentor.train()
        model_classifier.train()

        if param['loss_mode'] == 0:
            adj_sampled, adj_logits = model_augmentor(features, adj_norm, adj_orig)
            t_logits = model_classifier.teacher_head(model_classifier(adj_sampled, features))
            t_loss = nn.CrossEntropyLoss()(t_logits[train_mask], labels[train_mask])
            s_logits = model_classifier.student_head(model_classifier(adj_norm, features))
            s_loss = nn.CrossEntropyLoss()(s_logits[train_mask], labels[train_mask])
            ga_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            kd_loss = com_distillation_loss(t_logits, s_logits, adj_orig, adj_sampled, param['temp_kd'], param['loss_mode'])
            loss = param['ratio_ga'] * ga_loss + t_loss + s_loss + param['ratio_kd'] * kd_loss
        elif param['loss_mode'] == 1:
            adj_sampled, adj_logits = model_augmentor(features, adj_norm, adj_orig)
            t_logits = model_classifier.teacher_net(adj_sampled, features)
            t_loss = nn.CrossEntropyLoss()(t_logits[train_mask], labels[train_mask])
            s_logits = model_classifier.student_net(features)
            s_loss = nn.CrossEntropyLoss()(s_logits[train_mask], labels[train_mask])
            ga_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            kd_loss = com_distillation_loss(t_logits, s_logits, adj_orig, adj_sampled, param['temp_kd'], param['loss_mode'])
            loss = param['ratio_ga'] * ga_loss + t_loss + s_loss + param['ratio_kd'] * kd_loss
        elif param['loss_mode'] == 2:
            adj_sampled, adj_logits = model_augmentor(features, adj_norm, adj_orig)
            logits = model_classifier.teacher_net(adj_sampled, features)
            t_loss = nn.CrossEntropyLoss()(logits[train_mask], labels[train_mask])
            ga_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            loss = param['ratio_ga'] * ga_loss + t_loss
            s_loss = kd_loss = torch.zeros(1).to(device)
        elif param['loss_mode'] == 3:
            logits = model_classifier.teacher_net(adj_norm, features)
            loss = nn.CrossEntropyLoss()(logits[train_mask], labels[train_mask])
            ga_loss = s_loss = t_loss = kd_loss = torch.zeros(1).to(device)

        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()

        model_augmentor.eval()
        model_classifier.eval()
        if param['loss_mode'] == 0:
            logits_eval = model_classifier.student_head(model_classifier(adj_norm, features))
        elif param['loss_mode'] == 1:
            logits_eval = model_classifier.student_net(features)
        elif param['loss_mode'] == 2 or param['loss_mode'] == 3:
            logits_eval = model_classifier.teacher_net(adj_norm, features)
        train_acc = evaluation(logits_eval[train_mask], labels[train_mask])
        val_acc = evaluation(logits_eval[val_mask], labels[val_mask])
        test_acc = evaluation(logits_eval[test_mask], labels[test_mask])

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            es = 0
        else:
            es += 1
            if es == 200:
                print('Early stop!')
                break

        if epoch % 10 == 0:
            print('\033[0;30;46m Epoch [{:3}/{}]: GA Loss {:.6f}, Teacher Loss {:.6f}, Student Loss {:.6f}, KD Loss {:.6f} | Train Acc {:.4f}, Val Acc {:.4f}, Test Acc {:.4f} | {:.4f}, {:.4f}\033[0m'.format(
                epoch+1, param['n_epochs'], ga_loss.item(), t_loss.item(), s_loss.item(), kd_loss.item(), train_acc, val_acc, test_acc, test_val, test_best))

    return test_acc, test_val, test_best

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'citeseer', 'cornell', 'texas', 'wisconsin', 'film', 'squirrel', 'chameleon'])

    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temp_ga', type=float, default=1.0)
    parser.add_argument('--temp_kd', type=float, default=1.0)
    parser.add_argument('--ratio_ga', type=float, default=1.0)
    parser.add_argument('--ratio_kd', type=float, default=5.0)
   
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--ga_epochs', type=int, default=300)
    parser.add_argument('--nc_epochs', type=int, default=300)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--save_mode', type=int, default=1)
    parser.add_argument('--loss_mode', type=int, default=0)
    parser.add_argument('--data_mode', type=int, default=1)

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())

    if param['data_mode'] == 0:
        param['dataset'] = 'cora'
    if param['data_mode'] == 1:
        param['dataset'] = 'citeseer'
    if param['data_mode'] == 2:
        param['dataset'] = 'cornell'
    if param['data_mode'] == 3:
        param['dataset'] = 'texas'
    if param['data_mode'] == 4:
        param['dataset'] = 'wisconsin'
    if param['data_mode'] == 5:
        param['dataset'] = 'film'
    if param['data_mode'] == 6:
        param['dataset'] = 'squirrel'
    if param['data_mode'] == 7:
        param['dataset'] = 'chameleon'

    if os.path.exists("../param/best_parameters.json"):
        if param['loss_mode'] < 0:
            param = json.loads(open("../param/best_parameters.json", 'r').read())[param['dataset']]['0']

    if param['loss_mode'] == 3:
        param['ga_epochs'] = 0
        param['nc_epochs'] = 0
        param['n_epochs'] = 500

    features, adj_orig, adj_norm, labels, train_mask, val_mask, test_mask, nclass = load_data(param['dataset'])
    norm_w = adj_orig.shape[0]**2 / float((adj_orig.shape[0]**2 - adj_orig.sum()) * 2)
    pos_weight = torch.FloatTensor([float(adj_orig.shape[0]**2 - adj_orig.sum()) / adj_orig.sum()]).to(device)

    if param['save_mode'] == 0:
        SetSeed(param['seed'])
        test_acc, test_val, test_best = main()
        nni.report_final_result(test_val)

    else:
        test_acc_list = []
        test_val_list = []
        test_best_list = []

        for seed in range(5):
            SetSeed(seed + param['seed'] * 5)
            test_acc, test_val, test_best = main()
            test_acc_list.append(test_acc)
            test_val_list.append(test_val)
            test_best_list.append(test_best)
            nni.report_intermediate_result(test_val)

        nni.report_final_result(np.mean(test_val_list))
        outFile = open('../PerformMetrics.csv','a+', newline='')
        writer = csv.writer(outFile, dialect='excel')
        results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
        for v, k in param.items():
            results.append(k)
        
        results.append(str(test_acc_list))
        results.append(str(test_val_list))
        results.append(str(test_best_list))
        results.append(str(np.mean(test_acc_list)))
        results.append(str(np.mean(test_val_list)))
        results.append(str(np.mean(test_best_list)))
        results.append(str(np.std(test_acc_list)))
        results.append(str(np.std(test_val_list)))
        results.append(str(np.std(test_best_list)))
        writer.writerow(results)

