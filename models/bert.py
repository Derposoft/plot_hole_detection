"""
file containing the definitions for the baseline BERT models
that to find each of the 2 kinds of plot holes.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, aggr


class ContinuityBERT(nn.Module):
    """
    baseline model which finds Continuity Errors in plots --
    i.e., which sentences are plot holes and which ones are not.
    """
    def __init__(self, n_heads=16, input_dim=384, use_kg=False, kg_node_dim=100, kg_edge_dim=100):
        nn.Module.__init__(self)
        # decider decides which sentences are continuity errors
        self.decider = nn.Transformer(nhead=n_heads, d_model=input_dim, batch_first=True)
        # GAT which will use KG
        self.use_kg = use_kg
        self.gat = GATv2Conv(in_channels=kg_node_dim, out_channels=kg_node_dim, edge_dim=kg_edge_dim)
        self.aggregator = aggr.MeanAggregation()
        # project feature space to single probability
        self.proj = nn.Linear(input_dim if not self.use_kg else input_dim+kg_node_dim, 1)
        # softmax normalizes all proj outputs to find sentence
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, kgs=None):
        """
        :param x: sequence of sentence encodings from a story with shape (batch_size, seq_len, input_dim) 
        :param kgs: knowledge graphs represented as the following map and shapes:
            {
                "node_feats": (batch_size, n_nodes, kg_node_dim),
                "edge_indices": (batch_size, 2, n_edges),
                "edge_features": (batch_size, n_edges, kg_edge_dim)
            }
        :returns: sequence of logits for each sentence
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # obtain decider output
        x = self.decider(x, torch.zeros(x.shape))

        # if using kg, concatenate kg output to decider output
        if self.use_kg:
            x_kgs = []
            for i in range(batch_size):
                x_kg = self.gat(kgs["node_feats"][i], kgs["edge_indices"][i], kgs["edge_features"][i])
                x_kg = self.aggregator(x_kg)
                x_kgs.append(x_kg)
            x_kgs = torch.stack(x_kgs, dim=0).reshape([batch_size, -1])

            # copy gnn output for each kg once for each item in sequence to extend seq_dim
            x_kgs_stacked = []
            for _ in range(seq_len):
                x_kgs_stacked.append(x_kgs)
            x_kgs_stacked = torch.stack(x_kgs_stacked, dim=1)
            x = torch.concat([x, x_kgs_stacked], dim=-1)

        # pass all output into projection layer
        x = self.proj(x)
        x = x.reshape([x.shape[0], -1])
        return self.softmax(x)


class UnresolvedBERT(nn.Module):
    """
    baseline model which finds Unresolved Storyline Errors in plots --
    i.e., whether or not the story was cut short before the storyline 
    was resolved.
    """
    def __init__(self, n_heads=16, input_dim=384, use_kg=False, kg_node_dim=100, kg_edge_dim=100):
        nn.Module.__init__(self)
        # decider decides which sentences are most important in deciding how "incomplete" story is
        self.decider = nn.Transformer(nhead=n_heads, d_model=input_dim, batch_first=True)
        # GAT which will use KG
        self.use_kg = use_kg
        self.gat = GATv2Conv(in_channels=kg_node_dim, out_channels=kg_node_dim, edge_dim=kg_edge_dim)
        self.aggregator = aggr.MeanAggregation()
        # project feature space to single probability
        self.proj = nn.Linear(input_dim if not self.use_kg else input_dim+kg_node_dim, 1)
        # sigmoid function to determine percentage of story cut off
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, kgs=None):
        """
        :param x: sequence of sentence encodings from a story with shape (batch_size, seq_len, input_dim) 
        :param kgs: knowledge graphs represented as the following map and shapes:
            {
                "node_feats": (batch_size, n_nodes, kg_node_dim),
                "edge_indices": (batch_size, 2, n_edges),
                "edge_features": (batch_size, n_edges, kg_edge_dim)
            }
        :returns: single logit determining percentage of story that was left out
        """
        batch_size = x.shape[0]

        # obtain decider output
        x = self.decider(x, torch.zeros([x.shape[0], 1, x.shape[-1]]))

        # if using kg, concatenate kg output to decider output
        if self.use_kg:
            # generate gnn output for each kg
            x_kgs = []
            for i in range(batch_size):
                x_kg = self.gat(kgs["node_feats"][i], kgs["edge_indices"][i], kgs["edge_features"][i])
                x_kg = self.aggregator(x_kg)
                x_kgs.append(x_kg)
            x_kgs = torch.stack(x_kgs, dim=0)
            x = torch.concat([x, x_kgs], dim=-1)

        # pass all output into projection layer
        x = self.proj(x)
        x = x.reshape([x.shape[0]])
        return self.sigmoid(x)


if __name__ == "__main__":
    batch_size = 2
    seq_len = 100
    kg_node_dim = 32
    kg_edge_dim = 69
    n_nodes = 20
    n_edges = 40
    x = torch.rand([batch_size, seq_len, 384])
    x_kg = {
        "node_feats": torch.rand([batch_size, n_nodes, kg_node_dim]),
        "edge_indices": torch.randint(0, n_nodes, [batch_size, 2, n_edges]),
        "edge_features": torch.rand([batch_size, n_edges, kg_edge_dim])
    }
    
    """
    test ContinuityBERT model *without* KG
    """
    # model output
    continuity_model = ContinuityBERT()
    y_hat = continuity_model(x)
    # expected output
    y = torch.zeros((batch_size, seq_len))
    print(f"ContinuityBERT,noKG, output shape: {y_hat.shape}, expected shape: {y.shape}")

    """
    test ContinuityBERT model *with* KG
    """
    # model output
    continuity_model = ContinuityBERT(use_kg=True, kg_node_dim=kg_node_dim, kg_edge_dim=kg_edge_dim)
    y_hat = continuity_model(x, x_kg)
    # expected output
    y = torch.zeros((batch_size, seq_len))
    print(f"ContinuityBERT,KG, output shape: {y_hat.shape}, expected shape: {y.shape}")

    """
    test UnresolvedBERT model *without* KG
    """
    # model output
    unresolved_model = UnresolvedBERT()
    y_hat = unresolved_model(x)
    # expected output
    y = torch.zeros((batch_size))
    print(f"UnresolvedBERT,noKG, output shape: {y_hat.shape}, expected shape: {y.shape}")

    """
    test UnresolvedBERT model *with* KG
    """
    # model output
    unresolved_model = UnresolvedBERT(use_kg=True, kg_node_dim=kg_node_dim, kg_edge_dim=kg_edge_dim)
    y_hat = unresolved_model(x, x_kg)
    # expected output
    y = torch.zeros((batch_size))
    print(f"UnresolvedBERT,KG, output shape: {y_hat.shape}, expected shape: {y.shape}")
