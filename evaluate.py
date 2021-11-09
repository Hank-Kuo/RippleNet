import os
import argparse
import collections

import model.net as net
import model.data_loader as data_loader
import utils.load_data as load_data
import utils.utils as utils

import numpy as np
import torch
from torch.utils import data as torch_data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


def evaluation(params, model, data_generator):
    auc_list = []
    acc_list = []
    f1_list = []
    model.eval()

    for users, items, labels, memories_h, memories_r,memories_t in data_generator:

        items = items.to(params.device)
        labels = labels.to(params.device)
        memories_h = memories_h.permute(1, 0, 2).to(params.device)
        memories_r = memories_r.permute(1, 0, 2).to(params.device)
        memories_t = memories_t.permute(1, 0, 2).to(params.device)
        
        return_dict = model(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        f1 = f1_score(labels, predictions)

        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
    
    metrics_mean = {'auc': float(np.mean(auc_list)), 'acc': float(np.mean(acc_list)), 'f1':float(np.mean(f1_list)) }
    return metrics_mean


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=555, help="Seed value.")
parser.add_argument("--model_dir", default="./experiments/rippleNet-movie/plus_model", help="Path to model checkpoint (by default train from scratch).")
parser.add_argument("--model_type", default="plus_model", help="Path to model checkpoint (by default train from scratch).")

def get_model(params, model_type):
    model = {
        'base_model': net.RippleNet(params),
        'basic_model': net.RippleNet_basic(params),
        'plus_model': net.RippleNet_plus(params),
        'replace_model': net.RippleNet_replace(params),
        'head0_model': net.RippleNet_head0(params),
        'head1_model': net.RippleNet_head1(params),
        'head2_model': net.RippleNet_head2(params),
    }
    return model[model_type]

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
    test_generator = torch_data.DataLoader(test_set, batch_size=params.batch_size, drop_last=False)
    
    # model
    model = get_model(params, args.model_type)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), params.learning_rate)
    utils.load_checkpoint(checkpoint_dir, model, optimizer)
    best_model = model.to(params.device)
    best_model.eval()
    test_metrics = evaluation(params, best_model, test_generator)
    print('Eval: test auc: %.4f  acc: %.4f  f1: %.4f'% (test_metrics['auc'], test_metrics['acc'], test_metrics['f1']))
