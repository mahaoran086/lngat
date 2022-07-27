import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from datasets import get_planetoid_dataset
from train_eval import random_planetoid_splits, run
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WikipediaNetwork, Actor

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=True)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--heads', type=int, default=10)
parser.add_argument('--output_heads', type=int, default=1)
args = parser.parse_args()

def get_color(labels):
    colors=["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF", "#8B00FF"]#,"g","y","o"#根据情况自己改配色
    color=[]
    for i in range(len(labels)):
        if(labels[i] == 0):
            color.append(colors[0])
        elif(labels[i] == 1):
            color.append(colors[1])
        elif(labels[i] == 2):
            color.append(colors[2])
        elif(labels[i] == 3):
            color.append(colors[3])
        elif(labels[i] == 4):
            color.append(colors[4])
        elif(labels[i] == 5):
            color.append(colors[5])
        else:color.append(colors[6])
    # for i in range(100):
    #     color.append(colors[0])
    # for i in range(100,200):
    #     color.append(colors[1])
    # for i in range(200,300):
    #     color.append(colors[2])
    # for i in range(300,400):
    #     color.append(colors[3])
    # for i in range(400,500):
    #     color.append(colors[4])
    # for i in range(500, 600):
    #     color.append(colors[5])

    return color

def visual(x, y):
    X = data.x[:600]
    Y = data.y[:600]
    print(type(X))#必须是np.array
    X_embedded = TSNE(n_components=2,init="pca").fit_transform(X)
    print(X_embedded.shape)
    colors=get_color(Y)#配置点的颜色
    x=X_embedded[:,0]#横坐标
    y=X_embedded[:,1]#纵坐标
    plt.scatter(x, y, c=colors, linewidths=0.5, marker='o',edgecolors='k')
    plt.show()
    # plt.savefig("tsne.jpg")

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GATConv(dataset.num_features, args.hidden,
                             heads=args.heads, dropout=args.dropout)
        self.conv2 = GATConv(args.hidden * args.heads, dataset.num_classes,
                             heads=args.output_heads, concat=False,
                             dropout=args.dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
# dataset = Actor(root='./data/Actor')
data = dataset[0]
print('data:', data)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)
