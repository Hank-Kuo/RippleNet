import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self._parse_args(args)
    
    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        
    def forward(self, item_model):
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            h_emb_list.append(self.entity_emb(memories_h[i]))
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, item_embeddings = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)
        scores = self.predict(item_embeddings, o_list)
        return_dict = self._compute_loss(scores, labels, h_emb_list, t_emb_list, r_emb_list)
        return_dict["scores"] = scores

        return return_dict

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        base_loss = self.criterion(scores, labels.float())

        kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)
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
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=3)

            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded))

            # [batch_size, dim, 1]
            v = torch.unsqueeze(item_embeddings, dim=2)

            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))

            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1)

            item_embeddings = self._update_item_embedding(item_embeddings, o)
            o_list.append(o)
        return o_list, item_embeddings



    def predict(self, item_embeddings, o_list):
        y = o_list[-1] # batch_size x dim

        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
        return torch.sigmoid(scores)
