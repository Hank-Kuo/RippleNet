import argparse
import numpy as np
import model.data_loader as data_loader
from tqdm import tqdm
from train import train
import model.model as net

np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')

'''
# default settings for Book-Crossing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
'''

parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')


from torch.utils import data as torch_data
import torch

class Dataset(torch_data.Dataset):
    def __init__(self, data, ripple_set, args):
        self.args = args
        self.data = data
        self.ripple_set = ripple_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user = self.data[index, 0]
        items = self.data[index, 1]
        labels = self.data[index, 2]
        
        memories_h, memories_r, memories_t = [], [], []
        
        for i in range(self.args.n_hop):
            memories_h.append(self.ripple_set[user][i][0])
            memories_r.append(self.ripple_set[user][i][1])
            memories_t.append(self.ripple_set[user][i][2])
        return items, labels, torch.LongTensor(memories_h), torch.LongTensor(memories_r), torch.LongTensor(memories_t)


def evaluation(args, model, data_generator):
    start = 0
    auc_list = []
    acc_list = []
    model.eval()
    for items, labels, memories_h, memories_r,memories_t in data_generator:
        items = items.cuda()
        labels = labels.cuda()
        memories_h = memories_h.permute(1, 0, 2).cuda()
        memories_r = memories_r.permute(1, 0, 2).cuda()
        memories_t = memories_t.permute(1, 0, 2).cuda()
        auc, acc = model.evaluate(items, labels, memories_h, memories_r, memories_t)
        auc_list.append(auc)
        acc_list.append(acc)
    return float(np.mean(auc_list)), float(np.mean(acc_list))    



def test(args, data_info):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]
    
    train_set = Dataset(train_data, ripple_set, args)
    eval_set = Dataset(eval_data, ripple_set, args)
    test_set = Dataset(test_data, ripple_set, args)
    train_generator = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False,shuffle=True)
    eval_generator = torch_data.DataLoader(eval_set, batch_size=args.batch_size, drop_last=False)
    test_generator = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False)

    model = net.RippleNet(args, n_entity, n_relation)
    
    if args.use_cuda:
        model.cuda()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
    )
    

    start_epoch_id = 0 
    for epoch_id in range(start_epoch_id, args.n_epoch + 1):
        print("Epoch {}/{}".format(epoch_id, args.n_epoch))
        loss_impacting_samples_count = 0
        samples_count = 0
        model.train()

        with tqdm(total=len(train_generator)) as t:
            for items, labels, memories_h, memories_r,memories_t in train_generator:
                items = items.cuda()
                labels = labels.cuda()
                memories_h = memories_h.permute(1, 0, 2).cuda()
                memories_r = memories_r.permute(1, 0, 2).cuda()
                memories_t = memories_t.permute(1, 0, 2).cuda()
                return_dict = model(items, labels, memories_h, memories_r, memories_t)
                loss = return_dict["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_impacting_samples_count += loss.item()
                samples_count += items.size()[0]

                t.set_postfix(loss = loss_impacting_samples_count / samples_count * 100)
                t.update()
        train_auc, train_acc = evaluation(args, model, train_generator)
        eval_auc, eval_acc = evaluation(args, model, eval_generator)
        test_auc, test_acc = evaluation(args, model, test_generator)
        print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'% (epoch_id, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

args = parser.parse_args()
show_loss = False
data_info = data_loader.load_data(args)
test(args, data_info)