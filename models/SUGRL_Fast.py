import torch.nn as nn
import torch
import torch.nn.functional as F


def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)#, result


class SUGRL_Fast(nn.Module):
    def __init__(self, n_in ,cfg = None, dropout = 0.2,sparse = True,adj_nums=0):
        super(SUGRL_Fast, self).__init__()
        self.MLP = make_mlplayers(n_in, cfg)
        self.dropout = dropout
        self.A = None
        self.sparse = sparse
        self.cfg = cfg

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq_a, adj_list=None):
        if self.A is None:
            self.A = adj_list
        seq_a = F.dropout(seq_a, self.dropout, training=self.training)

        h_a = self.MLP(seq_a)
        h_p_0 = F.dropout(h_a, self.dropout, training=self.training)

        h_p_list = [] # shape [(num_node,feature_dim),....,(num_node,feature_dim)]
        for adj in adj_list:
            if self.sparse:
                h_p = torch.spmm(adj, h_p_0)
                h_p_list.append(h_p)
            else:
                h_p = torch.mm(adj, h_p_0)
                h_p_list.append(h_p)

        # simple average
        h_p_list_unsqu = [ls.unsqueeze(0) for ls in h_p_list]
        h_p_fusion = torch.mean(torch.cat(h_p_list_unsqu), 0)

        return h_a, h_p_list, h_p_fusion

    def embed(self,  seq_a , adj_list=None ):
        h_a = self.MLP(seq_a)
        h_list = []
        for adj in adj_list:
            if self.sparse:
                h_p = torch.spmm(adj, h_a)
                h_list.append(h_p)
            else:
                h_p = torch.mm(adj, h_a)
                h_list.append(h_p)

        # simple average
        h_p_list_unsqu = [ls.unsqueeze(0) for ls in h_list]
        h_fusion = torch.mean(torch.cat(h_p_list_unsqu), 0)

        return h_a.detach(), h_p.detach(), h_fusion.detach()
