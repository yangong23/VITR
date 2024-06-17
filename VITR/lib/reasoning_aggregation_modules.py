import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def l1norm(X, dim, eps=1e-8):
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim=-1, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class EncoderReasoningAggregation(nn.Module):

    def __init__(self, embed_size, sim_dim, bert_size, thre_cat):
        super(EncoderReasoningAggregation, self).__init__()

        self.thre_cat = thre_cat
        self.rr_w = nn.Linear(embed_size, sim_dim)
        self.clip_w = nn.Linear(bert_size, sim_dim)

        in_channel = 49

        # Relational graph filter
        self.RelationalGraphFilter1 = RelationalGraphFilter(in_channels=in_channel, in_spatials=embed_size)
        self.RelationalGraphFilter2 = RelationalGraphFilter(in_channels=in_channel, in_spatials=embed_size)
        self.RelationalGraphFilter3 = RelationalGraphFilter(in_channels=in_channel, in_spatials=embed_size)
        self.RelationalGraphFilter4 = RelationalGraphFilter(in_channels=in_channel, in_spatials=embed_size)

        self.sim_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # GRU
        self.rnn = nn.GRU(sim_dim, sim_dim, 1, batch_first=True)

        # GNN
        self.GraphAggregating1 = GraphAggregating(sim_dim)
        self.GraphAggregating2 = GraphAggregating(sim_dim)
        self.GraphAggregating3 = GraphAggregating(sim_dim)
        self.GraphAggregating4 = GraphAggregating(sim_dim)

        self.init_weights()

    def forward(self, epoch, img_emb, img_embg, cap_emb, bemb, cap_lens, cap_lens2):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        # Relational graph filter
        Att_img_emb = img_emb
        Att_img_emb = self.RelationalGraphFilter1(Att_img_emb)
        Att_img_emb = self.RelationalGraphFilter2(Att_img_emb)
        Att_img_emb = self.RelationalGraphFilter3(Att_img_emb)
        Att_img_emb = self.RelationalGraphFilter4(Att_img_emb)
        img_emb = Att_img_emb
        bemb = l2norm(bemb, dim=-1)
        img_embg = l2norm(img_embg, dim=-1)

        for i in range(n_caption):
            # get the i-th description
            n_word = int(cap_lens[i].item())
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_glob = bemb[i, :, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            cap_i_expand_glob = cap_i_glob.repeat(n_image, 1, 1)

            # Text-image reasoner
            Context_img = TextImageReasoner(cap_i_expand, img_emb, smooth=12.0)

            # joining the local representations
            sim_rr = torch.pow(torch.sub(cap_i_expand, Context_img), 2)
            sim_rr = self.rr_w(sim_rr)
            sim_rr = l1norm(sim_rr, dim=-1)

            # joining the global representations
            sim_glob = torch.pow(torch.sub(cap_i_expand_glob, img_embg), 2)
            sim_glob = self.clip_w(sim_glob)
            sim_glob = l1norm(sim_glob, dim=-1)


            # condition for balancing the learning
            if epoch>=self.thre_cat:
                sim_emb = torch.cat([sim_glob, sim_rr], 1)
            else:
                sim_emb = sim_rr

            # GNN Aggregating
            sim_emb = self.GraphAggregating1(sim_emb)
            # sim_emb = l2norm(sim_emb, dim=-1)
            sim_emb = self.GraphAggregating2(sim_emb)
            # sim_emb = l2norm(sim_emb, dim=-1)
            sim_emb = self.GraphAggregating3(sim_emb)
            # sim_emb = l2norm(sim_emb, dim=-1)

            # GRU pooling
            out, hidden = self.rnn(sim_emb)
            sim_emb = hidden[0]
            # sim_emb = l2norm(sim_emb, dim=-1)

            # predict similarity score
            sim_i = self.sigmoid(self.sim_w(sim_emb))
            sim_all.append(sim_i)

        sim_all = torch.cat(sim_all, 1)

        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def TextImageReasoner(query, context, smooth, eps=1e-8):

    queryT = torch.transpose(query, 1, 2)

    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = F.softmax(attn*smooth, dim=2)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    contextT = torch.transpose(context, 1, 2)
    weightedContext = torch.bmm(contextT, attnT)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext

class RelationalGraphFilter(nn.Module):

    def __init__(self, in_channels, in_spatials, bn_layer=True):
        super(RelationalGraphFilter, self).__init__()

        self.in_channels = in_channels
        self.in_spatials = in_spatials

        activate = nn.Tanh()

        self.W1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels * 2, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.in_channels),
            activate
        )

        self.W2 = nn.Sequential(
            nn.Linear(self.in_spatials, 1),
            activate
        )

        self.W3 = nn.Sequential(
            nn.Linear(self.in_channels + 1, 1),
            activate
        )

        self.theta = nn.Sequential(
            nn.Linear(self.in_spatials, self.in_spatials),
            activate
        )
        self.phi = nn.Sequential(
            nn.Linear(self.in_spatials, self.in_spatials),
            activate
        )

        self.init_weights()

    def forward(self, v):
        theta_v = self.theta(v)
        phi_v = self.phi(v)
        phi_v = phi_v.permute(0, 2, 1)
        Gs = torch.matmul(theta_v, phi_v)

        Gs_in = Gs.permute(0, 2, 1)
        Gs_out = Gs
        Gs_joint = torch.cat((Gs_in, Gs_out), 1)
        Gs_joint = self.W1(Gs_joint)

        g_xs = self.W2(v)
        ys = torch.cat((g_xs, Gs_joint), 2)

        W_ys = self.W3(ys)

        W_ysEx = W_ys.expand_as(v)
        at = torch.sigmoid(W_ysEx)

        out = at * v

        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class GraphAggregating(nn.Module):
    def __init__(self, sim_dim):
        super(GraphAggregating, self).__init__()

        # Using Sequential with activation function
        self.query_w = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            # nn.Tanh()
        )
        self.key_w = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            # nn.Tanh()
        )
        self.sim_w = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            nn.Tanh()
        )

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.query_w(sim_emb)
        sim_key = self.key_w(sim_emb)
        sim_edge = torch.bmm(sim_query, sim_key.permute(0, 2, 1))
        sim_edge = torch.sigmoid(sim_edge)

        sim_graph = torch.bmm(sim_edge, sim_emb)
        sim_graph = self.sim_w(sim_graph) + sim_emb

        return sim_graph

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()


