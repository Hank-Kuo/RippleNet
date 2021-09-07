import torch
from torch.utils import data as torch_data

class Dataset(torch_data.Dataset):
    def __init__(self, params, data, ripple_set):
        self.params = params
        self.data = data
        self.ripple_set = ripple_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user = self.data[index, 0]
        items = self.data[index, 1]
        labels = self.data[index, 2]
        
        memories_h, memories_r, memories_t = [], [], []
        
        for i in range(self.params.n_hop):
            memories_h.append(self.ripple_set[user][i][0])
            memories_r.append(self.ripple_set[user][i][1])
            memories_t.append(self.ripple_set[user][i][2])
        return items, labels, torch.LongTensor(memories_h), torch.LongTensor(memories_r), torch.LongTensor(memories_t)
