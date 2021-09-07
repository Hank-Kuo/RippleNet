
def evaluation(params, model, data_generator):
    auc_list = []
    acc_list = []
    model.eval()
    for items, labels, memories_h, memories_r,memories_t in data_generator:
        items = items.to(params.device)
        labels = labels.to(params.device)
        memories_h = memories_h.permute(1, 0, 2).to(params.device)
        memories_r = memories_r.permute(1, 0, 2).to(params.device)
        memories_t = memories_t.permute(1, 0, 2).to(params.device)
        auc, acc = model.evaluate(items, labels, memories_h, memories_r, memories_t)
        auc_list.append(auc)
        acc_list.append(acc)
    return float(np.mean(auc_list)), float(np.mean(acc_list))    


