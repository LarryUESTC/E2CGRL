import time

from embedder import embedder
import os
from tqdm import tqdm
from evaluate import evaluate
from models.SUGRL_Fast import SUGRL_Fast
import numpy as np
import torch.nn.functional as F
import torch


import warnings
warnings.filterwarnings("ignore")

class E2_SGRL(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):

        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]

        print("Started training...")
        model = SUGRL_Fast(self.args.ft_size, cfg=self.cfg, dropout=0.2,sparse=self.args.sparse,adj_nums = self.args.adj_nums).to(self.args.device)
        my_margin = self.args.margin1
        my_margin_2 = my_margin + self.args.margin2
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
        lbl_z = torch.tensor([0.]).to(self.args.device)

        model.train()
        start = time.time()
        for _ in tqdm(range(self.args.nb_epochs)):
            optimiser.zero_grad()
            idx_list = []
            for i in range(self.args.neg_num):
                idx_0 = np.random.permutation(self.args.nb_nodes)
                idx_list.append(idx_0)

            h, h_pos_list, z = model(features, adj_list)

            """compute loss"""
            '''step 1:compute positive sample distance'''
            # compute distance between anchor and positive embeddings in Eq8 :d(h,h^+)^2
            s_pos_ls = []
            for i in range(len(h_pos_list)):
                s_pos_ls.append(F.pairwise_distance(h, h_pos_list[i]))

            # compute distance between  anchor and  common embedding in Eq16 :d(h,z)^2
            u_pos_fusion = F.pairwise_distance(h, z)

            '''step 2:compute negative sample distance'''
            # compute distance between anchor and negative embeddings in Eq8 :d(h,h^-)^2
            s_neg_ls = []
            for negative_id in idx_list:
                s_neg_ls.append(F.pairwise_distance(h, h[negative_id]))

            margin_label = -1 * torch.ones_like(s_pos_ls[0])

            loss_s = 0
            loss_u = 0
            loss_c = 0

            '''step 3:compute loss'''
            # compute L_s Eq8 and 17
            for i in range(len(s_pos_ls)):
                for s_neg in s_neg_ls:
                    loss_s += (margin_loss(s_pos_ls[i], s_neg, margin_label)).mean()
            # compute L_c Eq16 and L_u Eq11
            for s_neg in s_neg_ls:
                loss_c += (margin_loss(u_pos_fusion, s_neg, margin_label)).mean()
                loss_u += torch.max((s_neg - u_pos_fusion.detach() - my_margin_2), lbl_z).sum()
            loss_u = loss_u / self.args.neg_num

            loss = loss_s * self.args.w_s \
                   + loss_u * self.args.w_u \
                   + loss_c * self.args.w_c

            loss.backward()
            optimiser.step()

        training_time = time.time() - start
        print("training time:{}s".format(training_time))
        print("Evaluating...")
        model.eval()
        ha, hp, hf = model.embed(features, adj_list)
        macro_f1s, micro_f1s, k1, k2, st = evaluate(hf, self.idx_train, self.idx_val, self.idx_test, self.labels,
                                                    seed=self.args.seed, epoch=self.args.test_epo,
                                                    lr=self.args.test_lr)  # ,seed=seed
        return macro_f1s, micro_f1s, k1, k2, st, training_time
