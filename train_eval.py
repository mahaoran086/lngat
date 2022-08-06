from __future__ import division

import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# device = 'cpu'


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None):

    val_losses, accs, durations = [], [], []
    for _ in range(runs):
        data = dataset[0]

        # #
        # data.train_mask = data.train_mask[:, 0]
        # data.val_mask = data.val_mask[:, 0]
        # data.test_mask = data.test_mask[:, 0]
        # #

        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes)
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []
        # print(eval_info['val_loss'])

        for epoch in range(1, epochs + 1):

            train(model, optimizer, data)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch
            #print(eval_info['val_loss'])

            if logger is not None:
                logger(eval_info)

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']

                test_acc = eval_info['test_acc']
                # ##
                # result = eval_info['test_pred'].eq(eval_info['test_data.y']).cpu().tolist()
                # for i in range(len(result)):
                #     if result[i] is False:
                #         index.append(i + 1708)
                # ##

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break




        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)



    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    print('acc:', acc)

    print('Val Loss: {:.4f}, Test Accuracy: {:.4f} ± {:.4f}, MAX Accuracy: {:.4f}, Duration: {:.3f}'.
          format(loss.mean().item(),
                 acc.mean().item(),     ## mean（）
                 acc.std().item(),
                 acc.max().item(),
                 duration.mean().item()))


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    # print(data.train_mask)
    # print(out[data.train_mask])
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    index = []

    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()

        pred = logits[mask].max(1)[1]

        # print("+++++++", pred.size(), data.x[data['test_mask']].size())
        # torch.save(data.x[data['test_mask']].to(torch.device('cpu')), "x.pth")
        # torch.save(pred.to(torch.device('cpu')), "y.pth")

        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

        outs['{}_pred'.format(key)] = pred
        outs['{}_data.y'.format(key)] = data.y[mask]

        # if key == 'test':
        #     result = pred.eq(data.y[mask]).cpu().tolist()
        #     print(pred.eq(data.y[mask]).sum())
        #     for i in range(len(result)):
        #         if result[i] is False:
        #             index.append(i+1708)

                # if pred[i].eq(data.y[mask][i]):
                #  index.append(i)
            # print('logits[mask]', logits[mask], logits[mask].size())
            # print('logits[mask].max(1)', logits[mask].max(1)[1])
            # a = torch.Tensor(index)
            # a = a.cpu()
            # b = a.detach().numpy()
            # np.savetxt("./index.csv", b)

            # print('pred:', pred.size())
            # print('data.y[mask]:', data.y[mask])
            # print('data[mask]:', data['test_mask'])

    return outs
