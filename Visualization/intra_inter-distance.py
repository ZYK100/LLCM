import scipy.io
import torch
import numpy as np
#import time
import os
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import scipy.io

######################################################################
result = scipy.io.loadmat('tsne.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_label = torch.FloatTensor(result['query_label'][0])
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = torch.FloatTensor(result['gallery_label'][0])

query_feature = query_feature.detach().cpu().numpy()
gallery_feature = gallery_feature.detach().cpu().numpy()


mask = query_label.expand(len(gallery_label), len(query_label)).eq(gallery_label.expand(len(query_label), len(gallery_label)).t()).cuda()

distmat = torch.FloatTensor(1 - np.matmul(gallery_feature, np.transpose(query_feature))).cuda() #Cosine distance
intra = distmat[mask]
inter = distmat[mask == 0]

######################################################################
plt.rcParams.update({'font.size': 14})


fig, ax = plt.subplots()
b = np.linspace(0.3, 1.3, num=1000)

ax.hist(intra.detach().cpu().numpy(), b, histtype="stepfilled", alpha=0.6, color = 'blue', density=True, label='Intra-class')
ax.hist(inter.detach().cpu().numpy(), b, histtype="stepfilled", alpha=0.6, color = 'green', density=True, label='Inter-class')

ax.set_xlabel('Feature Distance')
ax.set_ylabel('Frequency')
ax.legend()


fig.savefig('scatter.svg',dpi=1000,format='svg')
plt.show()
