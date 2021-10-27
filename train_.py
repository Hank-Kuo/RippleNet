import os
import argparse
import json
from tqdm import tqdm
import logging

import model.net as net
from evaluate import evaluation
import model.data_loader as data_loader
import utils.load_data as load_data
import utils.utils as utils
import utils.tensorboard as tensorboard

import numpy as np
import torch
from torch.utils import data as torch_data

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=555, help="Seed value.")
parser.add_argument("--model_dir", default="./experiments/rippleNet-movie/base_model", help="Path to model checkpoint (by default train from scratch).")
parser.add_argument("--model_type", default="base", help="Path to model checkpoint (by default train from scratch).")

def main():
    args = parser.parse_args()
    
    # torch setting
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # os setting
    params_path = os.path.join(args.model_dir, 'params.json')
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoint')
    test_best_json_path = os.path.join(args.model_dir, "metrics_test_best_weights.json")

    # params
    params = utils.Params(params_path)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    
    # load dataset
    print("===> Loading datasets")
    train_data, test_data, n_entity, n_relation, max_user_history_item, ripple_set= load_data.load_data(params)
    params.n_entity = n_entity
    params.n_relation = n_relation
    params.max_user_history_item = max_user_history_item

    # data loader
    train_set = data_loader.Dataset(params, train_data, ripple_set)
    test_set = data_loader.Dataset(params, test_data, ripple_set)
    train_generator = torch_data.DataLoader(train_set, batch_size=params.batch_size, drop_last=False,shuffle=True)
    test_generator = torch_data.DataLoader(test_set, batch_size=params.batch_size, drop_last=False)
    
    # model
    print("===> Building model")
    model = net.RippleNet(params)
    
    model = model.to(params.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), params.learning_rate)
    tb = tensorboard.Tensorboard(args.model_dir, False)
    writer = tb.create_writer()
    start_epoch_id = 1
    step = 0
    best_score = 0.0

    logging.info("Number of Entity: {}, Number of Relation: {}, User History item: {}".format(n_entity, n_relation, max_user_history_item))
    logging.info("Training Dataset: {}, Test Dataset: {}".format(len(train_set), len(test_set)))
    logging.info("===> Starting training ...")
    logging.info(model.__class__.__name__)
    print(model)

    # Train
    for epoch_id in range(start_epoch_id, params.epochs + 1):
        print("Epoch {}/{}".format(epoch_id, params.epochs))
        
        loss_impacting_samples_count = 0
        samples_count = 0
        model.train()

        with tqdm(total=len(train_generator)) as t:
            for items, labels, memories_h, memories_r,memories_t in train_generator:
                items = items.to(params.device)
                labels = labels.to(params.device)
                memories_h = memories_h.permute(1, 0, 2).to(params.device)
                memories_r = memories_r.permute(1, 0, 2).to(params.device)
                memories_t = memories_t.permute(1, 0, 2).to(params.device)
                return_dict = model(items, labels, memories_h, memories_r, memories_t)
                loss = return_dict["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_impacting_samples_count += loss.item()
                samples_count += items.size()[0]
                step += 1

                t.set_postfix(loss = loss_impacting_samples_count / samples_count * 100)
                t.update()
                
                writer.add_scalar('Loss/base_loss', return_dict["loss"].data.cpu().numpy(), global_step=step)
                writer.add_scalar('Loss/kge_loss', return_dict["kge_loss"].data.cpu().numpy(), global_step=step)
                writer.add_scalar('Loss/l2_loss', return_dict["l2_loss"].data.cpu().numpy(), global_step=step)
                
       
        # validation
        if epoch_id % params.valid_every == 0:
            model.eval()

            test_metrics = evaluation(params, model, test_generator)
            logging.info('- Eval: test auc: %.4f  acc: %.4f  f1: %.4f'% (test_metrics['auc'], test_metrics['acc'], test_metrics['f1']))
            
            writer.add_scalar('Accuracy/test/AUC', test_metrics['auc'] , global_step=epoch_id)
            writer.add_scalar('Accuracy/test/ACC', test_metrics['acc'] , global_step=epoch_id)
            writer.add_scalar('Accuracy/test/F1', test_metrics['f1'] , global_step=epoch_id)
            
            score = test_metrics['auc']
            if score > best_score:
                best_score = score   
                test_metrics['epoch'] = epoch_id   
                       
                utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_id, best_score)
                utils.save_dict_to_json(test_metrics, test_best_json_path)
                    
    tb.finalize()

if __name__ == '__main__':
    main()