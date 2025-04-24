import torch
from torch_geometric.data import HeteroData


def reweight_hetero_graph( graph : HeteroData , triplet : tuple , weight):
    if not graph[triplet[0],triplet[1],triplet[2]]:
        return graph
    list_of_weights = graph[triplet[0],triplet[1],triplet[2]].edge_attr.tolist()
    for i in range(len(list_of_weights)):
        list_of_weights[i] = weight
    graph[triplet[0],triplet[1],triplet[2]].edge_attr = torch.tensor(list_of_weights, dtype=torch.float32)
    return graph
