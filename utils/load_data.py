import collections
import random
import os
import argparse
import numpy as np

import utils as utils

MAX_SAMPLE_TRIPLET = 15
MAX_HISTORY_ITEM = 10

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


def get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')
    print(user_history_dict[1])
  
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
                # tails_of_last_hop = ripple_set[user][-1][2]

            # search graph
            for entity in tails_of_last_hop:
                triplets = kg[entity]
                if len(triplets) < MAX_SAMPLE_TRIPLET:
                    sample_triplets = triplets
                else:
                    random.seed(555)
                    sample_triplets = random.sample(triplets, MAX_SAMPLE_TRIPLET)
                
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