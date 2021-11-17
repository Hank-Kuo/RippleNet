import collections
import random
import os
import argparse
import numpy as np
from numpy import linalg as LA
import json

import utils.utils as utils

MAX_SAMPLE_TRIPLET =10#15
MAX_HISTORY_ITEM = 10#10

def load_data(args):
    rating_file = './data/' + args.dataset + '/ratings_final'
    kg_file = './data/' + args.dataset + '/kg_final'

    train_data, test_data, user_history_dict = load_rating(args, rating_file)
    n_entity, n_relation, kg = load_kg(args, kg_file)
    max_user_history_item, ripple_set = get_ripple_set(args, kg, user_history_dict)

    return train_data, test_data, n_entity, n_relation, max_user_history_item, ripple_set


def load_rating(args, rating_file):
    print('reading rating file ...')

    # reading rating file
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    return dataset_split(rating_np)


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:user_history:test = 6:2:2
    train_ratio = 0.6
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]
    np.random.seed(555)
    train_indices = np.random.choice(n_ratings, size=int(n_ratings * train_ratio), replace=False)
    left = set(range(n_ratings)) - set(train_indices)
    np.random.seed(555)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    user_history_indices = list(left - set(test_indices))

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in user_history_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            if len(user_history_dict[user]) < MAX_HISTORY_ITEM:
                user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]

    train_data = rating_np[train_indices]
    test_data = rating_np[test_indices]


    return train_data, test_data, user_history_dict


def load_kg(args, kg_file):
    print('reading KG file ...')

    # reading kg file
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg

def load_weight():
    entity_np = np.load('./data/weight/entity.npy')
    relation_np = np.load('./data/weight/relation.npy')
    with open('./data/weight/entity.dict') as f:
        entity_dict = json.load(f)
    with open('./data/weight/relation.dict') as f:
        relation_dict = json.load(f)
    return entity_np, relation_np, entity_dict, relation_dict



def get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')
    entity_np, relation_np, entity_dict, relation_dict = load_weight()
  
    max_user_history_item = MAX_HISTORY_ITEM* MAX_SAMPLE_TRIPLET
    ripple_set = collections.defaultdict(list)
    
    for user in user_history_dict:
        temp_set = set()
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []
            
            # find seed to search graph
            if h == 0:
                tails_of_last_hop = user_history_dict[user]
                temp_set = set(tails_of_last_hop)
            else:
                tails_of_last_hop = list(set(ripple_set[user][-1][2]) - temp_set)
                temp_set = set.union(temp_set, set(ripple_set[user][-1][2]))
                
            temp_score = []
            
            # search graph
            for entity in tails_of_last_hop:
                triplets = kg[entity]
                temp = []
                for index, tail_and_relation in enumerate(triplets):
                    if str(entity) in entity_dict:
                        entity_id =  entity_dict[str(entity)]
                    else:
                        entity_id = 0
                    if str(tail_and_relation[0]) in entity_dict:
                        tail_id = entity_dict[str(tail_and_relation[0])]
                    else:
                        tail_id = 0
                   
                    if str(tail_and_relation[1]) in relation_dict:
                        relation_id = relation_dict[str(tail_and_relation[1])]
                    else:
                        relation_id = 0
                   
                    score = LA.norm(entity_np[entity_id]+ relation_np[relation_id]-entity_np[tail_id], 1)
                    temp.append(score) 
                
                temp_idx = np.argsort(temp)
                for i, v in enumerate(temp_idx):
                    if v < MAX_SAMPLE_TRIPLET:
                        temp_score.append(temp[v])
                        memories_h.append(entity)
                        memories_r.append(triplets[i][1])
                        memories_t.append(triplets[i][0])

            # sampling
            if len(memories_h) == 0:
                ripple_set[user].append([[0]*max_user_history_item, [0]*max_user_history_item,[0]*max_user_history_item])
            else:    
                # padding sampling
                replace = len(memories_h) > max_user_history_item
                if replace == True:
                    sort_idx = np.argsort(temp_score)
                    indice = np.arange(len(memories_h))
                    indices = indice[sort_idx < max_user_history_item]
                else:
                    indices = np.arange(len(memories_h))
                # padding 
                l = max(0, max_user_history_item - len(memories_h))
                memories_h = [memories_h[i] for i in indices] + [0]*l
                memories_r = [memories_r[i] for i in indices] + [0]*l
                memories_t = [memories_t[i] for i in indices] + [0]*l
                
                ripple_set[user].append([memories_h, memories_r, memories_t])

    return max_user_history_item, ripple_set




if __name__ == '__main__':

    
    # torch setting
    np.random.seed(555)
    
    # os setting
    params_path = os.path.join('./experiments/base_model', 'params.json')
    params = utils.Params(params_path)

    train_data, test_data, n_entity, n_relation, max_user_history_item, ripple_set= load_data(params)