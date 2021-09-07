import argparse
from torchviz import make_dot
import model.model as net
import model.data_loader as data_loader
import torch

def get_feed_dict(args, data, ripple_set, start, end):
    items = torch.LongTensor(data[start:end, 1])
    labels = torch.LongTensor(data[start:end, 2])
    memories_h, memories_r, memories_t = [], [], []
    for i in range(args.n_hop):
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))
    if args.use_cuda:
        items = items.cuda()
        labels = labels.cuda()
        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))
        memories_t = list(map(lambda x: x.cuda(), memories_t))
    return items, labels, memories_h, memories_r,memories_t

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-100k', help='which dataset to use')
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

parser.add_argument('--use_cuda', type=bool, default=False, help='whether to use gpu')

args = parser.parse_args()
data_info = data_loader.load_data(args)

train_data = data_info[0]
eval_data = data_info[1]
test_data = data_info[2]
n_entity = data_info[3]
n_relation = data_info[4]
ripple_set = data_info[5]

items = torch.LongTensor([[2], [1]])
labels = torch.LongTensor([[0], [1]])
memories_h, memories_r, memories_t = [], [], []

model = net.RippleNet(args, n_entity, n_relation)
start = 0
item, label , h, r, t =  get_feed_dict(args, train_data, ripple_set, start, start + 2)
y = model(item, label , h, r, t )
MyConvNetVis = make_dot(y["scores"], params=dict(list(model.named_parameters())))
# MyConvNetVis.format = "png"
# 生成文件
MyConvNetVis.view()


params = list(model.parameters())
k = 0
for i in params:
    l = 1
    print("該層的結構：" + str(list(i.size())))
    for j in i.size():
            l *= j
    print("該層引數和：" + str(l))
    k = k + l
print("總引數數量和：" + str(k))