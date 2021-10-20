import os
import argparse

import model.net as net
import model.data_loader as data_loader
import utils.load_data as load_data
import utils.utils as utils
import utils.graph as graph

import numpy as np
import torch
from torch.utils import data as torch_data

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=555, help="Seed value.")
parser.add_argument("--model_dir", default="./experiments/rippleNet-movie/basic_model", help="Path to model checkpoint (by default train from scratch).")
parser.add_argument("--model_type", default="visual_model", help="Path to model checkpoint (by default train from scratch).")

def get_model(params, model_type):
    model = {
        'base_model': net.RippleNet(params),
        'basic_model': net.RippleNet_basic(params),
        'plus_model': net.RippleNet_plus(params),
        'replace_model': net.RippleNet_replace(params),
        'head0_model': net.RippleNet_head0(params),
        'head1_model': net.RippleNet_head1(params),
        'head2_model': net.RippleNet_head2(params),
        'visual_model': net.RippleNet_visual(params)
    }
    return model[model_type]

def get_ouput_np(batch, hop, memories_h, memories_r, memories_t, prob_list):
    head = memories_h[hop][batch]
    relation = memories_r[hop][batch]
    tail = memories_t[hop][batch]
    prob = prob_list[hop][batch]
    output = torch.stack((head, relation, tail, prob))
    output_np = output.detach().cpu().numpy()
    return output_np


if __name__ == '__main__':
    args = parser.parse_args()
    
    # torch setting
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
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

    # data loader
    test_set = data_loader.Dataset(params, test_data, ripple_set)
    test_generator = torch_data.DataLoader(test_set, batch_size=1024, drop_last=False)

    # model
    model = get_model(params, args.model_type)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), params.learning_rate)
    utils.load_checkpoint(checkpoint_dir, model, optimizer)
    best_model = model.to(params.device)
    best_model.eval()

    items, labels, memories_h, memories_r,memories_t = next(iter(test_generator))
    items = items.to(params.device)
    labels = labels.to(params.device)
    memories_h = memories_h.permute(1, 0, 2).to(params.device)
    memories_r = memories_r.permute(1, 0, 2).to(params.device)
    memories_t = memories_t.permute(1, 0, 2).to(params.device)
    return_dict, prob_list = model(items, labels, memories_h, memories_r, memories_t)
    
    batch = 11
    hop = 0
    output_np = get_ouput_np(batch, hop, memories_h, memories_r, memories_t, prob_list)
    output_np_1 = get_ouput_np(batch, hop+1, memories_h, memories_r, memories_t, prob_list)

    g = graph.PrintGraph()
    color_set = set()
    for i in range(180):
        h, r, t, w = output_np[:, i]
        if h > 0 and t >0 and abs(w)>0.4 :
            g.add_edge(int(h), int(t), round(w, 4))
            color_set.add(int(h))

    for i in range(180):
        h, r, t, w = output_np_1[:, i]
        if int(h) in g.G.nodes or int(t) in g.G.nodes:
            if h > 0 and t >0 and abs(w)>0.1:
                g.add_edge(int(h), int(t), round(w, 4))
                print('add')
    g.display(list(color_set))
        