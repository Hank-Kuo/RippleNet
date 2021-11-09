import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

class AKUPM(nn.Module):
    def __init__(self, args):
        super(AKUPM, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim, padding_idx=0)
        proj_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        beta_emb_1 = nn.Linear(self.dim, 1, bias=False)
        beta_emb_2 = nn.Linear(self.dim, 1, bias=False)

        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(proj_emb.weight)
        torch.nn.init.xavier_uniform_(beta_emb_1.weight)
        torch.nn.init.xavier_uniform_(beta_emb_2.weight)

        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.proj_emb = proj_emb
        self.beta_emb_1 = beta_emb_1
        self.beta_emb_2 = beta_emb_2

    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item

    def forward(self, items, labels, memories_h, memories_r, memories_t):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        proj_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            proj_emb_list.append(self.proj_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(scores, labels, h_emb_list, t_emb_list, r_emb_list)
        return_dict["scores"] = scores

        return return_dict

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        base_loss = self.criterion(scores, labels.float())

        kge_loss = 0
        for hop in range(self.n_hop):
            # [batch size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)
            # [batch size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)
            # [batch size, n_memory, dim, dim]
            hRt = torch.squeeze(torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded))
            kge_loss += torch.sigmoid(hRt).mean()
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)

    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, item_embeddings):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)

            # [batch_size, n_memory, dim]
            e = torch.squeeze(torch.matmul(r_emb_list[hop], t_expanded))
            # [batch_size, n_memory, n_memory]
            c = torch.matmul(e, e.permute(0, 2, 1))
            # [batch_size, n_memory, 1]
            new_c = self.beta_emb_1(c)
            # [batch_size, n_memory, 1]
            c_normalized = F.softmax(new_c, dim=1)
            # [batch_size, dim]
            o = (e * c_normalized).sum(dim=1)
            o_list.append(o)
        return o_list, item_embeddings

    def self_attention(self,  e):
        c = torch.matmul(e, e.permute(0, 2, 1))
        # [batch_size, n_memory, 1]
        new_c = self.beta_emb_1(c)
        # [batch_size, n_memory, 1]
        c_normalized = F.softmax(new_c, dim=1)
        # [batch_size, dim]
        o = (e * c_normalized).sum(dim=1)
        return o

    def predict(self, item_embeddings, o_list):
        # [batch_size, n_memory, dim]
        a = torch.cat(o_list, dim=1)
        # [batch_size, dim, 1]
        item_expanded = torch.unsqueeze(item_embeddings, dim=3)
        # [batch_size, n_memory, 1]
        prob = torch.matmul(a, item_expanded)
        # [batch_size, n_memory, 1]
        prob_normalized = F.softmax(prob, dim=1)
        user = (a * prob_normalized).sum(dim=1)
        scores = (item_embeddings * user).sum(dim=1)        
        return torch.sigmoid(scores)

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(labels, predictions)

        return auc, acc, f1
