import torch.nn as nn
from torch_geometric.nn import GATv2Conv


SENTENCE_ENCODER_DIM = {
    "all-MiniLM-L6-v2": 384,
    "paraphrase-albert-small-v2": 768,
    "word2vec": 300,
}


def get_model_size(model: nn.Module):
    return sum([x.numel() for x in model.parameters()])


def initialize_gnn(kg_node_dim, kg_edge_dim, n_layers, gnn_type="gatv2"):
    # ensure that the gnn selected supports both node and edge features
    supported_gnn_types = {
        "gatv2": GATv2Conv,
    }
    gnn_type = gnn_type.lower()
    assert gnn_type in supported_gnn_types, f"gnn must be one {supported_gnn_types}"
    gnn_layer = supported_gnn_types[gnn_type]
    return nn.ModuleList(
        [
            gnn_layer(
                in_channels=kg_node_dim, out_channels=kg_node_dim, edge_dim=kg_edge_dim
            )
            for _ in range(n_layers)
        ]
    )
