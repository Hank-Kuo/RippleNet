'''
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score


class RippleNet(object):
    def __init__(self, args, n_entity, n_relation):
        self._parse_args(args, n_entity, n_relation)
        self._build_inputs()
        self._build_embeddings()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops

    def _build_inputs(self):
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")
        self.memories_h = []
        self.memories_r = []
        self.memories_t = []

        for hop in range(self.n_hop):
            self.memories_h.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))
            self.memories_r.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))
            self.memories_t.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))

    def _build_embeddings(self):
        self.entity_emb_matrix = tf.get_variable(name="entity_emb_matrix", dtype=tf.float64,
                                                 shape=[self.n_entity, self.dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        self.relation_emb_matrix = tf.get_variable(name="relation_emb_matrix", dtype=tf.float64,
                                                   shape=[self.n_relation, self.dim, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())

    def _build_model(self):
        # transformation matrix for updating item embeddings at the end of each hop
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float64,
                                                initializer=tf.contrib.layers.xavier_initializer())

        # [batch size, dim]
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))

            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r[i]))

            # [batch size, n_memory, dim]
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

        o_list = self._key_addressing()

        self.scores = tf.squeeze(self.predict(self.item_embeddings, o_list))
        self.scores_normalized = tf.sigmoid(self.scores)

    def _key_addressing(self):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)

            # [batch_size, dim, 1]
            v = tf.expand_dims(self.item_embeddings, axis=2)

            # [batch_size, n_memory]
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)

            # [batch_size, n_memory]
            probs_normalized = tf.nn.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, dim]
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)

            self.item_embeddings = self.update_item_embedding(self.item_embeddings, o)
            o_list.append(o)
        return o_list

    def update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = tf.matmul(o, self.transform_matrix)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # [batch_size]
        scores = tf.reduce_sum(item_embeddings * y, axis=1)
        return scores

    def _build_loss(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        self.kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        self.kge_loss = -self.kge_weight * self.kge_loss

        self.l2_loss = 0
        for hop in range(self.n_hop):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
            if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
                self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss

    def _build_train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc
'''
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
        #self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        #self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)
        #self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        self.criterion = nn.BCELoss()

    def _init_emb(self):
        entity_emb = nn.Embedding(self.n_entity, self.dim)
        relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)
        transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        
        torch.nn.init.xavier_uniform(entity_emb.weight)
        torch.nn.init.xavier_uniform(relation_emb.weight)
        torch.nn.init.xavier_uniform(transform_matrix.weight)
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.transform_matrix = transform_matrix


    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops

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
            r_emb_list.append(
                self.relation_emb(memories_r[i]).view(
                    -1, self.n_memory, self.dim, self.dim
                )
            )
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, item_embeddings = self._key_addressing(
            h_emb_list, r_emb_list, t_emb_list, item_embeddings
        )
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
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
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
        return auc, acc
        