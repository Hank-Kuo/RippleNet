import collections
import random
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import utils as utils


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
    return dataset_split(args, rating_np)


def dataset_split(args, rating_np):
    print('splitting dataset ...')
    k = pd.DataFrame(rating_np, columns=['user', 'item', 'label'])

    value_counts = k[k['label']==1]['user'].value_counts()
    to_remove = value_counts[value_counts <= args.history_item].index
    df = k[~k['user'].isin(to_remove)]
    user_history = df[df['label']==1].groupby('user').apply(lambda x: x.sample(args.history_item,random_state=555))
    user_history_id = [y for x, y in user_history.index.tolist()]
    df = df.drop(user_history_id)

    user_history_dict = dict()
    for i, u in enumerate(list(set(user_history['user'].tolist()))):
        sub = user_history[user_history['user']==u]['item'].tolist()
        
        if u not in user_history_dict:
            user_history_dict[u] = []
        user_history_dict[u].extend(sub)

    train_data, test_data = train_test_split(df.to_numpy(), train_size=0.8, random_state=42)
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


def get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')
  
    max_user_history_item = args.history_item* args.sample_triplet
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
                # tails_of_last_hop = ripple_set[user][-1][2]

            # search graph
            for entity in tails_of_last_hop:
                triplets = kg[entity]
                if len(triplets) < args.sample_triplet:
                    sample_triplets = triplets
                else:
                    random.seed(555)
                    sample_triplets = random.sample(triplets, args.sample_triplet)
                
                for tail_and_relation in sample_triplets:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # sampling
            if len(memories_h) == 0:
                ripple_set[user].append([[0]*max_user_history_item, [0]*max_user_history_item,[0]*max_user_history_item])
            else:    
                # padding sampling
                replace = len(memories_h) > max_user_history_item
                if replace == True:
                    np.random.seed(555)
                    indices = np.random.choice(len(memories_h), size=max_user_history_item, replace=False)
                else:
                    np.random.seed(555)
                    indices = np.random.choice(len(memories_h), size=len(memories_h), replace=False)
                # padding 
                l = max(0, max_user_history_item - len(memories_h))
                memories_h = [memories_h[i] for i in indices] + [0]*l
                memories_r = [memories_r[i] for i in indices] + [0]*l
                memories_t = [memories_t[i] for i in indices] + [0]*l
                
                ripple_set[user].append([memories_h, memories_r, memories_t])

    return max_user_history_item, ripple_set



parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=555, help="Seed value.")
parser.add_argument("--model_dir", default="../experiments/rippleNet/base_model", help="Path to model checkpoint (by default train from scratch).")    

if __name__ == '__main__':
    args = parser.parse_args()
    
    # torch setting
    np.random.seed(args.seed)
    
    # os setting
    params_path = os.path.join(args.model_dir, 'params.json')
    params = utils.Params(params_path)

    train_data, test_data, n_entity, n_relation, max_user_history_item, ripple_set= load_data(params)