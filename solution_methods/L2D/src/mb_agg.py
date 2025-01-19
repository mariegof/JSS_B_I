"""
Mini-batch Aggregation Utilities for Graph Neural Networks

Goals:
- Aggregate observations across mini-batches for efficient GNN processing
- Handle sparse graph operations for batch processing
- Compute graph pooling matrices for message passing

Inputs:
- obs_mb (torch.Tensor): Batch of graph observations
- n_node (int): Number of nodes in each graph
- graph_pool_type (str): Type of pooling operation ('average' or 'sum')
- batch_size (torch.Size): Shape of the batch
- device (torch.device): Device for tensor operations

Outputs:
- adj_batch (torch.Tensor): Aggregated sparse adjacency tensor for the batch
- graph_pool (torch.Tensor): Sparse graph pooling matrix for the batch
"""
import torch


def aggr_obs(obs_mb, n_node):
    # obs_mb is [m, n_nodes_each_state, fea_dim], m is number of nodes in batch
    idxs = obs_mb.coalesce().indices()
    vals = obs_mb.coalesce().values()
    new_idx_row = idxs[1] + idxs[0] * n_node
    new_idx_col = idxs[2] + idxs[0] * n_node
    idx_mb = torch.stack((new_idx_row, new_idx_col))
    # print(idx_mb)
    # print(obs_mb.shape[0])
    adj_batch = torch.sparse_coo_tensor(indices=idx_mb,
                                         values=vals,
                                         size=torch.Size([obs_mb.shape[0] * n_node,
                                                          obs_mb.shape[0] * n_node]),
                                         ).to(obs_mb.device)

    # print('aggr_obs', adj_batch.shape)
    return adj_batch


def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    """
    Calculate graph pooling matrix

    Args:
        graph_pool_type: str, 'average' or 'sum'
        batch_size: torch.Size, shape of batch
        n_nodes: int, number of nodes in each graph
        device: torch.device, device    
    """
    # batch_size is the shape of batch
    # for graph pool sparse matrix
    if graph_pool_type == 'average':
        elem = torch.full(size=(batch_size[0]*n_nodes, 1),
                          fill_value=1 / n_nodes,
                          dtype=torch.float32,
                          device=device).view(-1)
    else:
        elem = torch.full(size=(batch_size[0] * n_nodes, 1),
                          fill_value=1,
                          dtype=torch.float32,
                          device=device).view(-1)
    idx_0 = torch.arange(start=0, end=batch_size[0],
                         device=device,
                         dtype=torch.long)
    # print(idx_0)
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size[0]*n_nodes, 1)).squeeze()

    idx_1 = torch.arange(start=0, end=n_nodes*batch_size[0],
                         device=device,
                         dtype=torch.long)
    idx = torch.stack((idx_0, idx_1))
    graph_pool = torch.sparse_coo_tensor(idx, elem,
                                          torch.Size([batch_size[0],
                                                      n_nodes*batch_size[0]])
                                          ).to(device)

    return graph_pool


'''
def aggr_obs(obs_mb, n_node):
    # obs_mb is [m, n_nodes_each_state, fea_dim], m is number of nodes in batch
    # if obs is padded
    # print(batch_obs[0])
    mb_size = obs_mb.shape
    if mb_size[-1] == 3:
        n_sample_in_batch = mb_size[0]
        # print(n_sample_in_batch)
        tensor_temp = torch.arange(start=0,
                                   end=n_sample_in_batch,
                                   dtype=torch.long,
                                   device=obs_mb.device).t().unsqueeze(dim=1)
        # print(tensor_temp)
        tensor_temp = tensor_temp.expand(-1, mb_size[-1])
        # print(tensor_temp)
        tensor_temp = tensor_temp.repeat(n_node, 1)
        # print(tensor_temp)
        tensor_temp = tensor_temp.sort(dim=0)[0]
        # print(tensor_temp)
        tensor_temp = tensor_temp * n_node
        return obs_mb.view(-1, mb_size[-1]) + tensor_temp
    # if obs is adj
    else:
        idxs = obs_mb.coalesce().indices()
        vals = obs_mb.coalesce().values()
        new_idx_row = idxs[1] + idxs[0] * n_node
        new_idx_col = idxs[2] + idxs[0] * n_node
        idx_mb = torch.stack((new_idx_row, new_idx_col))
        # print(idx_mb)
        # print(obs_mb.shape[0])
        adj_batch = torch.sparse.FloatTensor(indices=idx_mb,
                                             values=vals,
                                             size=torch.Size([obs_mb.shape[0] * n_node,
                                                              obs_mb.shape[0] * n_node]),
                                             ).to(obs_mb.device)
        return adj_batch
'''

if __name__ == '__main__':
    print('Go home.')