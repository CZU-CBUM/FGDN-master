

from src.utils import label_dirichlet_partition, parition_non_iid
import torch

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'stix'
labels = []
number_of_nodes = 10000
class_num = 2
for i in range(class_num):
    labels += [i] * (number_of_nodes // class_num )

labels = torch.tensor(labels)
n_trainer = 20
colors = ['#6495ED', '#FFD700']  

for iid_beta in [1, 10, 100, 10000]:
    node_partitions = label_dirichlet_partition(labels, len(labels), class_num, n_trainer, beta = iid_beta)

    distributions = []
    for i in range(n_trainer):
        distribution = []
        for j in range(class_num):
            distribution.append(torch.sum(labels[node_partitions[i]] == j) / len(node_partitions[i]))
        distributions.append(distribution)
    distributions = torch.tensor(distributions)

    ind = list(range(1, n_trainer + 1))

    plt.bar(ind, distributions[:, 0], color=colors[0])

    bottom_acc = distributions[:, 0].clone()
    for i in range(1, class_num):
        plt.bar(ind, distributions[:, i],
                     bottom = bottom_acc, color=colors[i % len(colors)])
        bottom_acc += distributions[:, i]
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Remove tick marks on x-axis
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)

    plt.xlabel("Local Model", fontsize=35)
    plt.ylabel("Label Distribution", fontsize=30)
    if iid_beta == 10000:
        plt.title(f"i.i.d   $\\chi={iid_beta}$", fontsize=35)
    else:
        plt.title(f"Non-i.i.d   $\\chi={iid_beta}$", fontsize=35)
    plt.show()
