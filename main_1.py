import os
import argparse
from tqdm import tqdm

import model.net as net
from evaluate import evaluate
import model.data_loader as data_loader
import utils.load_data as load_data
import utils.utils as utils

import numpy as np
import torch
from torch.utils import data as torch_data

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=555, help="Seed value.")
parser.add_argument("--dataset", default="ml-100k", help="Which dataset used.")
parser.add_argument("--dataset_dir", default="./data", help="Path to dataset.")
parser.add_argument("--model_dir", default="./experiments/base_model", help="Path to model checkpoint (by default train from scratch).")


def main():
    args = parser.parse_args()
    
    # torch setting
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # os setting
    params_path = os.path.join(args.model_dir, 'params.json')
    dataset_path = os.path.join(args.dataset_dir, args.dataset) 
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoint')
    
    # params
    params = utils.Params(params_path)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    
    # load dataset
    print("===> Loading datasets")
    train_data, eval_data, test_data, n_entity, n_relation, ripple_set= load_data.LoadData(params)
 
    # data loader
    train_set = data_loader.Dataset(params, train_data, ripple_set)
    eval_set = data_loader.Dataset(params, eval_data, ripple_set)
    test_set = data_loader.Dataset(params, test_data, ripple_set)
    train_generator = torch_data.DataLoader(train_set, batch_size=params.batch_size, drop_last=False,shuffle=True)
    eval_generator = torch_data.DataLoader(eval_set, batch_size=params.batch_size, drop_last=False)
    test_generator = torch_data.DataLoader(test_set, batch_size=params.batch_size, drop_last=False)
    
    # model
    print("===> Building model")
    model = net.RippleNet(params, n_entity, n_relation)
    
    model = model.to(params.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), params.learning_rate)
    summary_writer = tensorboard.SummaryWriter(log_dir=args.model_dir)
    start_epoch_id = 1
    step = 0
    best_score = 0.0

    print("Training Dataset: {}".format(len(train_set)))
    print("Eval Dataset: {}".format(len(eval_set)))
    print("Test Dataset: {}".format(len(test_set)))
    print(model)
    
    logging.info("Starting training: {}...".format(args.dataset))
    
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

                t.set_postfix(loss = loss_impacting_samples_count / samples_count * 100)
                t.update()

                #summary_writer.add_scalar('Loss/train', loss.mean().data.cpu().numpy(), global_step=step)
                #summary_writer.add_scalar('Distance/positive', pd.sum().data.cpu().numpy(), global_step=step)
                #summary_writer.add_scalar('Distance/negative', nd.sum().data.cpu().numpy(), global_step=step)
       
        # validation
        if epoch_id % params.validation_freq == 0:
            model.eval()
            train_auc, train_acc = evaluation(params, model, train_generator)
            eval_auc, eval_acc = evaluation(params, model, eval_generator)
            test_auc, test_acc = evaluation(params, model, test_generator)
            logging.info('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'% (epoch_id, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))
            score = eval_auc
            if score > best_score:
                best_score = score
                utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_id, best_score)


if __name__ == '__main__':
    main()