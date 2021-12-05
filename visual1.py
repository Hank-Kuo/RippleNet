import os

import model.data_loader as data_loader
import utils.load_data as load_data
import utils.utils as utils
from train1 import get_model

import numpy as np
import pandas as pd
import torch
from pyvis.network import Network
from torch.utils import data as torch_data

model_dir = './experiments/rippleNet-movie-1m-wiki/base_model'
model_type = 'base_model'
seed = 555

def get_mapping():
    entity2idx_file = './data/' + 'movie-1m-wiki' + '/entity2idx.txt'
    relation2idx_file = './data/' + 'movie-1m-wiki' + '/relation2idx.txt'
    kg_full_file = './data/' + 'movie-1m-wiki' + '/movie-full.kg'

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
    idx2entity[0] ='PAD'
    idx2relation[0] ='PAD'
    return idx2entity, idx2relation

def get_ouput_df(batch, params, memories_h, memories_r, memories_t, prob_list):
    output = []
    user_history = params.max_user_history_item
    
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
    
    df = pd.DataFrame(o_np, columns=["head_name", 'relation_name', 'tail_name', 'weight'])
    df['isHistory'] = pd.Series(isHistory)
    print(df.shape)
    return df

def save_graph(df, idx2entity, idx2relation):
    
    got_net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=True)

    # set the physics layout of the network
    got_net.barnes_hut()

    sources = df['head_id']
    targets = df['tail_id']
    relation = df['relation_id']
    weight = df['weight']
    isHistory = df['isHistory']

    edge_data = zip(sources, targets, relation, weight, isHistory)

    for e in edge_data:
        src = idx2entity[e[0]]
        dst = idx2entity[e[1]]
        re_label =idx2relation[e[2]] 
        w = re_label +'('+str(e[3])+')'
        isCheck = str(e[4])

        
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
    got_net.save_graph('./visual/base_model.html')

np.random.seed(seed)
torch.random.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params_path = os.path.join(model_dir, 'params.json')
checkpoint_dir = os.path.join(model_dir, 'checkpoint')

# params
params = utils.Params(params_path)
params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_data, test_data, n_entity, n_relation, max_user_history_item, ripple_set= load_data.load_data(params)
params.n_entity = n_entity
params.n_relation = n_relation
params.max_user_history_item = max_user_history_item

# data loader
test_set = data_loader.Dataset(params, test_data, ripple_set)
test_generator = torch_data.DataLoader(test_set, batch_size=1024, drop_last=False)

model = get_model(params, model_type)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), params.learning_rate)
utils.load_checkpoint(checkpoint_dir, model, optimizer)
best_model = model.to(params.device)
best_model.eval()

user, items, labels, memories_h, memories_r,memories_t = next(iter(test_generator))
items = items.to(params.device)
labels = labels.to(params.device)
memories_h = memories_h.permute(1, 0, 2).to(params.device)
memories_r = memories_r.permute(1, 0, 2).to(params.device)
memories_t = memories_t.permute(1, 0, 2).to(params.device)
return_dict, prob_list = model(items, labels, memories_h, memories_r, memories_t)

idx2entity, idx2relation = get_mapping()

user, items, labels, memories_h, memories_r,memories_t = next(iter(test_generator))
items = items.to(params.device)
labels = labels.to(params.device)
memories_h = memories_h.permute(1, 0, 2).to(params.device)
memories_r = memories_r.permute(1, 0, 2).to(params.device)
memories_t = memories_t.permute(1, 0, 2).to(params.device)
return_dict, prob_list = model(items, labels, memories_h, memories_r, memories_t)

batch = 50
df = get_ouput_df(batch, params, memories_h, memories_r, memories_t, prob_list)
df['head_id'] = df['head_name'].astype(int)
df['tail_id'] = df['tail_name'].astype(int)
df['relation_id'] = df['relation_name'].astype(int)
df = df[['head_id','tail_id','relation_id', 'weight','isHistory']]
df['weight'] =  (df['weight']).map('{:,.4f}'.format)
print(items[batch], return_dict['scores'][batch], labels[batch])
save_graph(df, idx2entity, idx2relation)
