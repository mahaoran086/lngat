import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import LNGATConv, LNGATConv1
from torch_geometric.datasets import WikipediaNetwork, Actor
from datasets import get_planetoid_dataset
from train_eval import random_planetoid_splits, run



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=True)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()


class Net(torch.nn.Module):

    def __init__(self, dataset):
        super(Net, self).__init__()
        self.mlp = Seq(dataset.num_features, args.hidden)
        self.conv1 = LNGATConv(args.hidden, dataset.num_classes)
        self.conv2 = LNGATConv1(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x0 = F.relu(self.mlp(x))

        x1 = F.relu(self.conv1(x0, edge_index))
        x1 = F.dropout(x1, p=args.dropout, training=self.training)
        x2 = self.conv2(x0, edge_index)

        if use_global_pooling:
            x3 = torch.max(x0, axis=-1, keepdims=True)
            lngal_out = torch.cat((x1, x2, x3), axis=-1)

        lngal_out = lngal_out + x

        return F.log_softmax(lngal_out, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
# dataset = Actor(root='./data/Actor')
data = dataset[0]

permute_masks = random_planetoid_splits if args.random_splits else None

run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)







