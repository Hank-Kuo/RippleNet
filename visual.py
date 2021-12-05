import os
import argparse
import collections

import model.net as net
import model.data_loader as data_loader
import utils.load_data as load_data
import utils.utils as utils
from train import get_model

import numpy as np
import torch
import pandas as pd
from pyvis.network import Network
from torch.utils import data as torch_data

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=555, help="Seed value.")
parser.add_argument("--model_dir", default="./experiments/rippleNet-movie-1m-wiki/basic_model", help="Path to model checkpoint (by default train from scratch).")
parser.add_argument("--model_type", default="base_model", help="Path to model checkpoint (by default train from scratch).")

def get_mapping(params):
    entity2idx_file = './data/' + params.dataset + '/entity2idx.txt'
    relation2idx_file = './data/' + params.dataset+ '/relation2idx.txt'
    kg_full_file = './data/' + params.dataset + '/movie-full.kg'

    df  = pd.read_csv(kg_full_file, sep='\t')
    e_df  = pd.read_csv(entity2idx_file, sep='\t', names=['entity', 'id'])
    r_df  = pd.read_csv(relation2idx_file, sep='\t', names=['relation', 'id'])

    tail_df = df[['tail', 'tail_name']]
    tail_df.columns = ['entity', 'entity_name']
    head_df = df[['head', 'head_name']]
    head_df.columns = ['entity', 'entity_name']
    entity_df = pd.concat([head_df, tail_df]).drop_duplicates()
    relation_df = df[['relation', 'relation_name']]

    e_df2 = pd.merge(e_df, entity_df, on='entity')
    r_df2 = pd.merge(r_df, relation_df, on='relation')

    idx2entity =  dict([(a, b) for a, b in zip(e_df2['id'], e_df2['entity_name'])])
    idx2relation =  dict([(a, b) for a, b in zip(r_df2['id'], r_df2['relation'])])

    return idx2entity, idx2relation


def get_ouput_df(batch, params, memories_h, memories_r, memories_t, prob_list):
    output = []
    user_history = 10 * params.sample_triplet
    
    for hop in range(params.n_hop):
        head = memories_h[hop][batch]
        relation = memories_r[hop][batch]
        tail = memories_t[hop][batch]
        prob = prob_list[hop][batch]
        o = torch.stack((head, relation, tail, prob))
        output_np = o.detach().cpu().numpy()
        output.append(output_np)

    o_np = np.concatenate(output, axis=1).T

    isHistory = [True]*user_history + [False]*(o_np.shape[0] -user_history)
    
    df = pd.DataFrame(o_np, columns=["head", 'relation', 'tail', 'weight'])
    df['isHistory'] = pd.Series(isHistory)
    print(df.shape)
    return df


def id_map_name():
    print('213')
def save_graph(df):
    got_net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=True)

    # set the physics layout of the network
    got_net.barnes_hut()

    sources = df['head_name']
    targets = df['tail_name']
    #relation_label = df['relation_name']
    weight = df['weight']
    isHistory = df['isHistory']

    edge_data = zip(sources, targets, weight, isHistory)

    for e in edge_data:
        src = str(e[0])
        dst = str(e[1])
        #re_label = e[2]
        w = str(e[2])
        isCheck = str(e[3])

        
        if bool(isCheck)== True:
            got_net.add_node(src, src, title=src, color='red')
        else:
            got_net.add_node(src, src, title=src)
        
        got_net.add_node(dst, dst, title=dst)
        got_net.add_edge(src, dst, label=w, color='blue')

    neighbor_map = got_net.get_adj_list()

    # add neighbor data to node hover data
    '''
    for node in got_net.nodes:
        node['title'] += ' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
        node['value'] = len(neighbor_map[node['id']])
    '''
    print('Save test.html to root folder...')
    got_net.save_graph('./test.html')

if __name__ == '__main__':
    args = parser.parse_args()
    
     # torch setting
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # os setting
    params_path = os.path.join(args.model_dir, 'params.json')
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoint')

    # params
    params = utils.Params(params_path)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # load dataset
    print("===> Loading datasets")
    _, test_data, n_entity, n_relation, max_user_history_item, ripple_set= load_data.load_data(params)
    params.n_entity = n_entity
    params.n_relation = n_relation
    params.max_user_history_item = max_user_history_item

    entity2idx_file = './data/' + params.dataset + '/entity2idx.txt'
    relation2idx_file = './data/' + params.dataset + '/relation2idx.txt'

    # data loader
    test_set = data_loader.Dataset(params, test_data, ripple_set)
    test_generator = torch_data.DataLoader(test_set, batch_size=1024, drop_last=False)

    # model
    model = get_model(params, args.model_type)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), params.learning_rate)
    utils.load_checkpoint(checkpoint_dir, model, optimizer)
    best_model = model.to(params.device)
    best_model.eval()

    items, labels, memories_h, memories_r, memories_t = next(iter(test_generator))
    items = items.to(params.device)
    labels = labels.to(params.device)
    memories_h = memories_h.permute(1, 0, 2).to(params.device)
    memories_r = memories_r.permute(1, 0, 2).to(params.device)
    memories_t = memories_t.permute(1, 0, 2).to(params.device)
    return_dict, prob_list = model(items, labels, memories_h, memories_r, memories_t)
    
    batch = 11
    hop = 0
    out_df  = get_ouput_df(batch, memories_h, memories_r, memories_t, prob_list)
    save_graph(out_df)

    entity2idx_file = './data/' + 'movie-1m-wiki' + '/entity2idx.txt'
    relation2idx_file = './data/' + 'movie-1m-wiki' + '/relation2idx.txt'
    kg_full_file = './data/' + 'movie-1m-wiki' + '/movie-full.kg'

    idx2entity = collections.defaultdict(str)
    idx2relation = collections.defaultdict(str)
    df  = pd.read_csv(kg_full_file, sep='\t')
    e_df  = pd.read_csv(entity2idx_file, sep='\t', names=['entity', 'id'])
    r_df  = pd.read_csv(relation2idx_file, sep='\t', names=['relation', 'id'])

    tail_df = df[['tail', 'tail_name']]
    tail_df.columns = ['entity', 'entity_name']
    head_df = df[['head', 'head_name']]
    head_df.columns = ['entity', 'entity_name']
    entity_df = pd.concat([head_df, tail_df]).drop_duplicates()
    relation_df = df[['relation', 'relation_name']]

    e_df2 = pd.merge(e_df, entity_df, on='entity')
    r_df2 = pd.merge(r_df, relation_df, on='relation')
    
    idx2entity =  dict([(a, b) for a, b in zip(e_df2['id'], e_df2['entity_name'])])
    idx2relation =  dict([(a, b) for a, b in zip(r_df2['id'], r_df2['relation'])])
