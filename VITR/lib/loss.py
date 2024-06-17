import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def torch_cosine_sim(a, b):
    # sc = torch.randn(a.size(0), b.size(0))
    c = a.mm(b.t())
    d = c.max(1)[0]

    one = torch.ones_like(d)
    d = torch.where(d == 0, one, d)

    sc = (c / d).t()
    if torch.cuda.is_available():
        sc = sc.cuda()

    return sc

class ContrastiveLossLSEH(nn.Module):

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLossLSEH, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation
        self.torchsim = torch_cosine_sim

    def max_violation_on(self):
        self.max_violation = True
        print('Max violation on.')

    def max_violation_off(self):
        self.max_violation = False
        print('Max violation off.')

    def forward(self, scores, im, s, ids, img_ids, dv, bemb):

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if torch.cuda.is_available():
            dv = dv.cuda()

        img_ids = np.array(img_ids)
        map = torch.ones(len(img_ids), len(img_ids))
        for i in range(len(img_ids)):
            for j in range(len(img_ids)):
                if img_ids[i] == img_ids[j]:
                    map[i, j] = 0
        if torch.cuda.is_available():
            map = map.cuda()

        m_alpha = 0.185
        m_lambda = 0.025
        SeScores = self.torchsim(dv, dv)
        SeScores = SeScores * m_lambda
        SeMargin = SeScores + m_alpha
        SeMargin = SeMargin * map

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        SeMargin = SeMargin.masked_fill_(I, 0)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (SeMargin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (SeMargin + scores - d2).clamp(min=0)

        # clear diagonals
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum(), cost_s.sum(), cost_im.sum()


