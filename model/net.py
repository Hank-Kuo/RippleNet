import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

class RippleNet(nn.Module):
    def __init__(self, args):
        super(RippleNet, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
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
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, item_embeddings = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(labels, predictions)

        return auc, acc, f1
        
class RippleNet_basic(nn.Module):
    def __init__(self, args):
        super(RippleNet_basic, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
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
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(labels, predictions)

        return auc, acc, f1

class RippleNet_replace(nn.Module):
    def __init__(self, args):
        super(RippleNet_replace, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.item_update_mode = 'replace'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(labels, predictions)

        return auc, acc, f1

class RippleNet_plus(nn.Module):
    def __init__(self, args):
        super(RippleNet_plus, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.item_update_mode = 'plus'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(labels, predictions)

        return auc, acc, f1

class RippleNet_plus2(nn.Module):
    def __init__(self, args):
        super(RippleNet_plus2, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
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
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(labels, predictions)

        return auc, acc, f1

class RippleNet_head0_plus2(nn.Module):
    def __init__(self, args):
        super(RippleNet_head0_plus2, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
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
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        o_emb_0 = h_emb_list[0].mean(dim=1)
        o_list.append(o_emb_0)
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
        y = o_list[0] # batch_size x dim

        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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
        
class RippleNet_head1_plus2(nn.Module):
    def __init__(self, args):
        super(RippleNet_head1_plus2, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
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
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        o_emb_0 = h_emb_list[0].mean(dim=1)
        o_list.append(o_emb_0)
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
        y = 0 # batch_size x dim

        for i in range(self.n_hop - 1):
            y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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
        
class RippleNet_head2_plus2(nn.Module):
    def __init__(self, args):
        super(RippleNet_head2_plus2, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
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
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        
        o_emb_0 = h_emb_list[0].mean(dim=1)
        o_list.append(o_emb_0)

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

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(labels, predictions)

        return auc, acc, f1
       
class RippleNet_head0_replace(nn.Module):
    def __init__(self, args):
        super(RippleNet_head0_replace, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.item_update_mode = 'replace'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        o_emb_0 = h_emb_list[0].mean(dim=1)
        o_list.append(o_emb_0)
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
        y = o_list[0] # batch_size x dim

        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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
        
class RippleNet_head1_replace(nn.Module):
    def __init__(self, args):
        super(RippleNet_head1_replace, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.item_update_mode = 'replace'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        o_emb_0 = h_emb_list[0].mean(dim=1)
        o_list.append(o_emb_0)
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
        y = 0 # batch_size x dim

        for i in range(self.n_hop - 1):
            y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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
        
class RippleNet_head2_replace(nn.Module):
    def __init__(self, args):
        super(RippleNet_head2_replace, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.item_update_mode = 'replace'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        
        o_emb_0 = h_emb_list[0].mean(dim=1)
        o_list.append(o_emb_0)

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

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(labels, predictions)

        return auc, acc, f1
     
class RippleNet_head0_att_plus2(nn.Module):
    def __init__(self, args):
        super(RippleNet_head0_att_plus2, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
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
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        
        # item att
        alpha = 0 
        
        # [batch_size, dim, 1]
        item = torch.unsqueeze(item_embeddings, dim=2) 
        # [batch_size, n_memory]
        item_att = torch.squeeze(torch.matmul(h_emb_list[0], item)) # * alpha 
        # [batch_size, n_memory, 1]
        item_att_normal = torch.unsqueeze(F.softmax(item_att, dim=1), dim=2) 
        # [batch_size, dim]
        o_emb_0 = (h_emb_list[0] * item_att_normal).sum(dim=1) 
        
        o_list.append(o_emb_0)

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
        y = o_list[0]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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
        
class RippleNet_head1_att_plus2(nn.Module):
    def __init__(self, args):
        super(RippleNet_head1_att_plus2, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
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
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        
        # item att
        alpha = 0 
        # [batch_size, dim, 1]
        item = torch.unsqueeze(item_embeddings, dim=2) 
        # [batch_size, n_memory]
        item_att = torch.squeeze(torch.matmul(h_emb_list[0], item)) # * alpha 
        # [batch_size, n_memory, 1]
        item_att_normal = torch.unsqueeze(F.softmax(item_att, dim=1), dim=2) 
        # [batch_size, dim]
        o_emb_0 = (h_emb_list[0] * item_att_normal).sum(dim=1) 
        o_list.append(o_emb_0)

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
        y = 0

        for i in range(self.n_hop - 1):
            y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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
        
class RippleNet_head2_att_plus2(nn.Module):
    def __init__(self, args):
        super(RippleNet_head2_att_plus2, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
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
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        
        # item att
        alpha = 0         
        # [batch_size, dim, 1]
        item = torch.unsqueeze(item_embeddings, dim=2) 
        # [batch_size, n_memory]
        item_att = torch.squeeze(torch.matmul(h_emb_list[0], item)) # * alpha 
        # [batch_size, n_memory, 1]
        item_att_normal = torch.unsqueeze(F.softmax(item_att, dim=1), dim=2) 
        # [batch_size, dim]
        o_emb_0 = (h_emb_list[0] * item_att_normal).sum(dim=1) 
        o_list.append(o_emb_0)

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
            for i in range(self.n_hop):
                y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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

class RippleNet_head2_att_plus2_1(nn.Module):
    def __init__(self, args):
        super(RippleNet_head2_att_plus2_1, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
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
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        
        # item att
        alpha = 0         
        # [batch_size, dim, 1]
        item = torch.unsqueeze(item_embeddings, dim=2) 
        # [batch_size, n_memory]
        item_att = torch.squeeze(torch.matmul(h_emb_list[0], item)) # * alpha 
        # [batch_size, n_memory, 1]
        item_att_normal = torch.unsqueeze(F.softmax(item_att, dim=1), dim=2) 
        # [batch_size, dim]
        o_emb_0 = (h_emb_list[0] * item_att_normal).sum(dim=1) 
        o_list.append(o_emb_0)
        item_embeddings = self.transform_matrix(item_embeddings + o_emb_0)

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
            for i in range(self.n_hop):
                y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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

class RippleNet_head0_att_replace(nn.Module):
    def __init__(self, args):
        super(RippleNet_head0_att_replace, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.item_update_mode = 'replace'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        
        # item att
        alpha = 0 
        # [batch_size, dim, 1]
        item = torch.unsqueeze(item_embeddings, dim=2) 
        # [batch_size, n_memory]
        item_att = torch.squeeze(torch.matmul(h_emb_list[0], item)) # * alpha 
        # [batch_size, n_memory, 1]
        item_att_normal = torch.unsqueeze(F.softmax(item_att, dim=1), dim=2) 
        # [batch_size, dim]
        o_emb_0 = (h_emb_list[0] * item_att_normal).sum(dim=1) 
        
        o_list.append(o_emb_0)

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
        y = o_list[0]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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
        
class RippleNet_head1_att_replace(nn.Module):
    def __init__(self, args):
        super(RippleNet_head1_att_replace, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.item_update_mode = 'replace'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        
        # item att
        alpha = 0 
        # [batch_size, dim, 1]
        item = torch.unsqueeze(item_embeddings, dim=2) 
        # [batch_size, n_memory]
        item_att = torch.squeeze(torch.matmul(h_emb_list[0], item)) # * alpha 
        # [batch_size, n_memory, 1]
        item_att_normal = torch.unsqueeze(F.softmax(item_att, dim=1), dim=2) 
        # [batch_size, dim]
        o_emb_0 = (h_emb_list[0] * item_att_normal).sum(dim=1) 
        o_list.append(o_emb_0)

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
        y = 0

        for i in range(self.n_hop - 1):
            y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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
        
class RippleNet_head2_att_replace(nn.Module):
    def __init__(self, args):
        super(RippleNet_head2_att_replace, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.item_update_mode = 'replace'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        
        # item att
        alpha = 0         
        # [batch_size, dim, 1]
        item = torch.unsqueeze(item_embeddings, dim=2) 
        # [batch_size, n_memory]
        item_att = torch.squeeze(torch.matmul(h_emb_list[0], item)) # * alpha 
        # [batch_size, n_memory, 1]
        item_att_normal = torch.unsqueeze(F.softmax(item_att, dim=1), dim=2) 
        # [batch_size, dim]
        o_emb_0 = (h_emb_list[0] * item_att_normal).sum(dim=1) 
        o_list.append(o_emb_0)
        # item_embeddings = o_emb_0

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
            for i in range(self.n_hop):
                y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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
        
class RippleNet_head2_att_replace_1(nn.Module):
    def __init__(self, args):
        super(RippleNet_head2_att_replace_1, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.item_update_mode = 'replace'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        
        # item att
        alpha = 0         
        # [batch_size, dim, 1]
        item = torch.unsqueeze(item_embeddings, dim=2) 
        # [batch_size, n_memory]
        item_att = torch.squeeze(torch.matmul(h_emb_list[0], item)) # * alpha 
        # [batch_size, n_memory, 1]
        item_att_normal = torch.unsqueeze(F.softmax(item_att, dim=1), dim=2) 
        # [batch_size, dim]
        o_emb_0 = (h_emb_list[0] * item_att_normal).sum(dim=1) 
        o_list.append(o_emb_0)
        item_embeddings = o_emb_0

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
            for i in range(self.n_hop):
                y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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


class RippleNet_head2_att_replace_1_gamma(nn.Module):
    def __init__(self, args):
        super(RippleNet_head2_att_replace_1_gamma, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.gamma = args.gamma
        self.item_update_mode = 'replace'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
               
        # [batch_size, dim, 1]
        item = torch.unsqueeze(item_embeddings, dim=2) 
        # [batch_size, n_memory]
        item_att = torch.squeeze(torch.matmul(h_emb_list[0], item)) * self.gamma 
        # [batch_size, n_memory, 1]
        item_att_normal = torch.unsqueeze(F.softmax(item_att, dim=1), dim=2) 
        # [batch_size, dim]
        o_emb_0 = (h_emb_list[0] * item_att_normal).sum(dim=1) 
        o_list.append(o_emb_0)
        item_embeddings = o_emb_0

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
            for i in range(self.n_hop):
                y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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
     

class RippleNet_no_relation(nn.Module):
    def __init__(self, args):
        super(RippleNet_no_relation, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.item_update_mode = 'replace'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
            v = torch.unsqueeze(item_embeddings, dim=2) 
            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(h_emb_list[hop], v))

            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1)

            item_embeddings = self._update_item_embedding(item_embeddings, o)
            o_list.append(o)
        return o_list, item_embeddings

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
            for i in range(self.n_hop):
                y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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

class RippleNet_no_relation_gamma(nn.Module):
    def __init__(self, args):
        super(RippleNet_no_relation_gamma, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
        self.gamma = args.gamma
        self.item_update_mode = 'replace'
        self.using_all_hops = True 

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, _ = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
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
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        # [batch_size, dim, 1]
        item = torch.unsqueeze(item_embeddings, dim=2) 
        # [batch_size, n_memory]
        item_att = torch.squeeze(torch.matmul(h_emb_list[0], item)) * self.gamma 
        # [batch_size, n_memory, 1]
        item_att_normal = torch.unsqueeze(F.softmax(item_att, dim=1), dim=2) 
        # [batch_size, dim]
        o_emb_0 = (h_emb_list[0] * item_att_normal).sum(dim=1) 
        o_list.append(o_emb_0)
        item_embeddings = o_emb_0
 
        for hop in range(self.n_hop):
            v = torch.unsqueeze(item_embeddings, dim=2) 
            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(h_emb_list[hop], v))

            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1)

            item_embeddings = self._update_item_embedding(item_embeddings, o)
            o_list.append(o)
        return o_list, item_embeddings

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
            for i in range(self.n_hop):
                y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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

class RippleNet_visual(nn.Module):
    def __init__(self, args):
        super(RippleNet_visual, self).__init__()

        self._parse_args(args)
        self._init_emb()
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity+1, self.dim, padding_idx=0)
        relation_emb = nn.Embedding(self.n_relation+1, self.dim * self.dim, padding_idx=0)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform_(entity_emb.weight)
        torch.nn.init.xavier_uniform_(relation_emb.weight)
        torch.nn.init.xavier_uniform_(transform_matrix.weight)
        
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.learning_rate = args.learning_rate
        self.n_memory = args.max_user_history_item
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
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, item_embeddings, prob_list = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)
        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
        return_dict["scores"] = scores

        return return_dict, prob_list

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        base_loss = self.criterion(scores, labels.float())

        kge_loss = 0
        for hop in range(self.n_hop):
            # [batch size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)
            # [batch size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)
            # [batch size, n_memory, dim, dim]
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
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
        prob_list = []
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
            prob_list.append(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1)

            item_embeddings = self._update_item_embedding(item_embeddings, o)
            o_list.append(o)
        return o_list, item_embeddings, prob_list

    def _update_item_embedding(self, item_embeddings, o):
        item_embeddings = self.transform_matrix(item_embeddings + o)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1] # batch_size x dim

        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]
        
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)        
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
        