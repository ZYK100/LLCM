import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable

class CPMLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(CPMLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=0.2)

    def forward(self, inputs, targets):
        ft1, ft2, ft3, ft4 = torch.chunk(inputs, 4, 0)
        lb1, lb2, lb3, lb4 = torch.chunk(targets, 4, 0)
        
        lb_num = len(lb1.unique())    
        lbs = lb1.unique() 

        n = lbs.size(0)   

        ft1 = ft1.chunk(lb_num, 0)
        ft2 = ft2.chunk(lb_num, 0)
        ft3 = ft3.chunk(lb_num, 0)
        ft4 = ft4.chunk(lb_num, 0)
        center1 = []
        center2 = []
        center3 = []
        center4 = []
        for i in range(lb_num):
            center1.append(torch.mean(ft1[i], dim=0, keepdim=True))
            center2.append(torch.mean(ft2[i], dim=0, keepdim=True))
            center3.append(torch.mean(ft3[i], dim=0, keepdim=True))
            center4.append(torch.mean(ft4[i], dim=0, keepdim=True))

        ft1 = torch.cat(center1)
        ft2 = torch.cat(center2)
        ft3 = torch.cat(center3)
        ft4 = torch.cat(center4)

        dist_13 = pdist_torch(ft1, ft3)
        dist_23 = pdist_torch(ft2, ft3)
        dist_33 = pdist_torch(ft3, ft3)
        dist_11 = pdist_torch(ft1, ft1)

        dist_14 = pdist_torch(ft1, ft4)
        dist_24 = pdist_torch(ft2, ft4)
        dist_44 = pdist_torch(ft4, ft4)
        dist_22 = pdist_torch(ft2, ft2)

        mask = lbs.expand(n, n).eq(lbs.expand(n, n).t())
        
        dist_ap_123, dist_an_123, dist_ap_124, dist_an_124, dist_an_33, dist_an_44, dist_an_11, dist_an_22 = [], [], [], [], [], [], [], []
        for i in range(n):
            dist_ap_123.append(dist_23[i][mask[i]].max().unsqueeze(0))
            dist_an_123.append(dist_13[i][mask[i]].min().unsqueeze(0))
            dist_an_33.append(dist_33[i][mask[i] == 0].min().unsqueeze(0))
            dist_an_11.append(dist_11[i][mask[i] == 0].min().unsqueeze(0))

            dist_ap_124.append(dist_14[i][mask[i]].max().unsqueeze(0))
            dist_an_124.append(dist_24[i][mask[i]].min().unsqueeze(0))
            dist_an_44.append(dist_44[i][mask[i] == 0].min().unsqueeze(0))
            dist_an_22.append(dist_22[i][mask[i] == 0].min().unsqueeze(0))

        dist_ap_123 = torch.cat(dist_ap_123)
        dist_an_123 = torch.cat(dist_an_123).detach()
        dist_an_33 = torch.cat(dist_an_33)
        dist_an_11 = torch.cat(dist_an_11)

        dist_ap_124 = torch.cat(dist_ap_124)
        dist_an_124 = torch.cat(dist_an_124).detach()
        dist_an_44 = torch.cat(dist_an_44)
        dist_an_22 = torch.cat(dist_an_22)

        loss_123 = self.ranking_loss(dist_an_123, dist_ap_123, torch.ones_like(dist_an_123)) + (self.ranking_loss(dist_an_33, dist_ap_123, torch.ones_like(dist_an_33)) + self.ranking_loss(dist_an_11, dist_ap_123, torch.ones_like(dist_an_33))) * 0.5
        loss_124 = self.ranking_loss(dist_an_124, dist_ap_124, torch.ones_like(dist_an_124)) + (self.ranking_loss(dist_an_44, dist_ap_124, torch.ones_like(dist_an_44)) + self.ranking_loss(dist_an_22, dist_ap_124, torch.ones_like(dist_an_44))) * 0.5
        return (loss_123 + loss_124)/2


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss

        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx