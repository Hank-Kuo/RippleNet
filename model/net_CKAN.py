import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score


class RippleNet(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(RippleNet, self).__init__()

        self._parse_args(args, n_entity, n_relation)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity, self.dim)
        relation_emb = nn.Embedding(self.n_relation, self.dim)
        attention = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
        )
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        for layer in attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.attention = attention


    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.n_memory
        self.item_update_mode = 'plus_transform'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        
        item_embeddings = self.entity_emb(items) # [batch size, dim]
        user_embeddings = [] # [n_hop, batch size, dim]
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb = self.entity_emb(memories_h[i])
            # [batch size, n_memory, dim]
            r_emb = self.relation_emb(memories_r[i])
            # [batch size, n_memory, dim]
            t_emb = self.entity_emb(memories_t[i])
            
            h_emb_list.append(h_emb)
            r_emb_list.append(r_emb)
            t_emb_list.append(t_emb)

            user_emb = self._knowledge_attention(h_emb, r_emb, t_emb)
            user_embeddings.append(user_emb)

        scores = self.predict(item_embeddings, user_embeddings)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
        return_dict["scores"] = scores

        return return_dict

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        base_loss = self.criterion(scores, labels.float())

        kge_loss = 0
        for hop in range(self.n_hop):
            score = h_emb_list[hop] + r_emb_list[hop] - t_emb_list[hop]
            kge_loss += torch.sigmoid(score).mean()
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)

    def _update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix(item_embeddings + o)
        elif self.item_update_mode == "no_update":
            item_embeddings = item_embeddings
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1] # batch_size x dim

        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)
        
        return torch.sigmoid(scores)

    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        # [batch_size, triple_set_size]
        att_weights = self.attention(torch.cat((h_emb,r_emb),dim=-1)).squeeze(-1)
        # [batch_size, triple_set_size]
        att_weights_norm = F.softmax(att_weights,dim=-1)
        # [batch_size, triple_set_size, dim]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        # [batch_size, dim]
        emb_i = emb_i.sum(dim=1)
        return emb_i

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc
        