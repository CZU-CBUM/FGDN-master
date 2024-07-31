# Construct the adjacency matrix of the population from phenotypic scores
import csv
import numpy as np
import os

import pandas as pd
import scipy as sp

# Input data variables
root_folder = ''
data_folder = os.path.join(root_folder, 'data/')
phenotype = os.path.join(root_folder, 'data/MDD841.csv')

# Make sure each site is represented in the training set when selecting a subset of the training set
def site_percentage(train_ind, perc, subject_list):
    """
        train_ind    : indices of the training samples
        perc         : percentage of training set used
        subject_list : list of subject IDs

    return:
        labeled_indices      : indices of the subset of training samples
    """

    train_list = subject_list[train_ind]
    sites = get_subject_score(train_list, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()
    site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])

    labeled_indices = []

    for i in np.unique(site):
        id_in_site = np.argwhere(site == i).flatten()

        num_nodes = len(id_in_site)
        labeled_num = int(round(perc * num_nodes))
        labeled_indices.extend(train_ind[id_in_site[:labeled_num]])

    return labeled_indices


def get_ids(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """

    mdd_sub_IDs = np.genfromtxt(os.path.join(data_folder, 'mdd_sub_IDs.txt'), dtype=str)

    if num_subjects is not None:
        mdd_sub_IDs = mdd_sub_IDs[:num_subjects]

    return mdd_sub_IDs

# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}
    with open('data/MDD841.csv', 'r', encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['ID'] in subject_list:
                scores_dict[row['ID']] = row[score]

    return scores_dict

# Construct the adjacency matrix of the population from phenotypic scores
def create_affinity_graph_from_scores(scores, subject_list):
    """
        scores       : list of phenotypic information to be used to construct the affinity graph
        subject_list : list of subject IDs

    return:
        graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = get_subject_score(subject_list, l)

        # quantitative phenotypic scores
        if l in ['AGE_AT_SCAN', 'EDU']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

mdd_sub_IDs = get_ids()
# Compute population graph using gender and acquisition site
graph = create_affinity_graph_from_scores(['SITE_ID', 'SEX', 'AGE_AT_SCAN'], mdd_sub_IDs)
#np.savetxt('adjacency_matrix.csv', graph, delimiter=',')


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, float(-1)).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


import torch
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)









