import math

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import precision_score
from torch.nn import init

import torch as th
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from torch.nn import init


class REConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 num_type=4,
                 weight=True,
                 bias=True,
                 activation=None):
        super(REConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.weight_type = nn.Parameter(th.ones(num_type))

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, type_info):
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')

            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = th.reshape(norm, shp)
                feat = feat * norm

            feat = th.matmul(feat, self.weight)
            graph.srcdata['h'] = feat * self.weight_type[type_info].reshape(-1, 1)
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class AGTLayer(nn.Module):
    def __init__(self, embeddings_dimension, nheads=2, att_dropout=0.5, emb_dropout=0.5, temper=1.0, rl=False, rl_dim=4,
                 beta=1):

        super(AGTLayer, self).__init__()

        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension

        self.head_dim = self.embeddings_dimension // self.nheads

        self.leaky = nn.LeakyReLU(0.01)

        self.temper = temper

        self.rl_dim = rl_dim

        self.beta = beta

        self.linear_l = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.linear_r = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)

        self.att_l = nn.Linear(self.head_dim, 1, bias=False)
        self.att_r = nn.Linear(self.head_dim, 1, bias=False)

        if rl:
            self.r_source = nn.Linear(rl_dim, rl_dim * self.nheads, bias=False)
            self.r_target = nn.Linear(rl_dim, rl_dim * self.nheads, bias=False)

        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.embeddings_dimension, bias=False)
        self.dropout1 = nn.Dropout(att_dropout)
        self.dropout2 = nn.Dropout(emb_dropout)

        self.LN = nn.LayerNorm(embeddings_dimension)

    def forward(self, h, rh=None):
        batch_size = h.size()[0]
        fl = self.linear_l(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        fr = self.linear_r(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)

        score = self.att_l(self.leaky(fl)) + self.att_r(self.leaky(fr)).permute(0, 1, 3, 2)

        if rh is not None:
            r_k = self.r_source(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).transpose(1, 2)
            r_q = self.r_target(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).permute(0, 2, 3, 1)
            score_r = r_k @ r_q
            score = score + self.beta * score_r

        score = score / self.temper

        score = F.softmax(score, dim=-1)
        score = self.dropout1(score)

        context = score @ fr

        h_sa = context.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.nheads)
        fh = self.linear_final(h_sa)
        fh = self.dropout2(fh)

        h = self.LN(h + fh)

        return h


class HINormer(nn.Module):
    def __init__(self, g, num_class, input_dimensions, embeddings_dimension=64, num_layers=8, num_gnns=2, nheads=2,
                 dropout=0, temper=1.0, num_type=4, beta=1):

        super(HINormer, self).__init__()

        self.g = g
        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.num_gnns = num_gnns
        self.nheads = nheads
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, embeddings_dimension) for in_dim in input_dimensions])

        self.dropout = dropout
        self.GCNLayers = torch.nn.ModuleList()
        self.RELayers = torch.nn.ModuleList()
        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_gnns):
            self.GCNLayers.append(GraphConv(
                self.embeddings_dimension, self.embeddings_dimension, activation=F.leaky_relu, allow_zero_in_degree=True))
            self.RELayers.append(REConv(num_type, num_type, activation=F.leaky_relu, num_type=num_type))
        for layer in range(self.num_layers):
            self.GTLayers.append(
                AGTLayer(self.embeddings_dimension, self.nheads, self.dropout, self.dropout, temper=temper, rl=True,
                         rl_dim=num_type, beta=beta))
        self.Drop = nn.Dropout(self.dropout)
        self.MeanPooling = nn.AvgPool2d((20, 1), stride=1)
        self.Prediction1 = nn.Linear(embeddings_dimension*2, embeddings_dimension)
        self.Prediction2 = nn.Linear(embeddings_dimension, num_class)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        # self.ReLU = nn.ReLU()

    def forward(self, features_list, query_batch, dataset_batch, type_emb, node_type, norm=True):
        h = []

        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        gh = torch.cat(h, 0)
        r = type_emb[node_type]
        for layer in range(self.num_gnns):
            gh = self.GCNLayers[layer](self.g, gh)
            gh = self.Drop(gh)
            r = self.RELayers[layer](self.g, r, node_type)
        # print(gh.shape, r.shape)
        # print('dataset_batch: ')
        # print(dataset_batch)
        dataset_batch = torch.LongTensor(np.array(dataset_batch))
        dh = gh[dataset_batch]
        ori_dh = gh[dataset_batch[:, 0]]
        dr = r[dataset_batch]
        # print('Dataset H shape: ')
        # print(dh.shape, dr.shape)
        for layer in range(self.num_layers):
            dh = self.GTLayers[layer](dh, rh=dr)
        # print(dh.shape)
        # print('Query Batch: ')
        qh_batch = []
        ori_qh = []
        for query_seqs in query_batch:
            # print(query_seqs)
            # query_seqs = torch.LongTensor(np.array(query_seqs))
            qh = gh[query_seqs]
            qr = r[query_seqs]
            for layer in range(self.num_layers):
                qh = self.GTLayers[layer](qh, rh=qr)
            qh = torch.unsqueeze(torch.mean(qh[:, 0, :], dim=0), 0)
            # qh = torch.unsqueeze(qh[:, 0, :].max(dim=0).values, 0)
            qh_batch.append(qh)
            ori_qh.append(torch.unsqueeze(torch.mean(gh[query_seqs[:, 0]], dim=0), 0))
            
        qh = torch.cat(qh_batch, 0)
        ori_qh = torch.cat(ori_qh, 0)

        dh = dh[:, 0, :]
        # qh = qh[:, 0, :]
        # ch = ch[:, 0, :]
        # print('Final H shape(d Q C): ')
        # print(dh.shape, qh.shape, ch.shape)
        input_embed = F.normalize(torch.cat((qh, dh), dim=1), dim=-1)
        # print(input_embed.shape)
        # output = self.Prediction(h[:, 0, :])
        # print(input_embed)
        logits = self.Prediction1(input_embed)
        logits = self.LeakyReLU(logits)
        logits = self.Prediction2(logits)
        # logits = nn.Sigmoid()(logits)
        # logits = logits / 0.5
        # if norm:
        #     logits = logits / (torch.norm(logits, dim=1, keepdim=True) + 1e-12)
        logits = torch.squeeze(logits, 1)
        # print(logits.shape)
        # logits = nn.functional.sigmoid(logits, dim=1)
        return logits
