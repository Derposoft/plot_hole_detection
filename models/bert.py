"""
file containing the definitions for the baseline BERT models
that to find each of the 2 kinds of plot holes.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, aggr
import models.model_utils as utils

device = "cuda" if torch.cuda.is_available() else "cpu"


class ContinuityBERT(nn.Module):  # ContinuityTransformer
    """
    baseline model which finds Continuity Errors in plots --
    i.e., which sentences are plot holes and which ones are not.
    """

    def __init__(
        self,
        n_heads=16,
        n_layers=6,
        n_gnn_layers=2,
        input_dim=384,
        hidden_dim=20,
        use_kg=False,
        kg_node_dim=100,
        kg_edge_dim=100,
        dropout=0.1,
    ):
        nn.Module.__init__(self)
        # embed into hidden dim
        full_hidden_dim = hidden_dim * n_heads
        self.embedder = nn.Linear(input_dim, full_hidden_dim)
        # decider decides which sentences are continuity errors
        self.decider = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                full_hidden_dim,
                n_heads,
                dim_feedforward=full_hidden_dim,
                batch_first=True,
                dropout=dropout,
            ),
            n_layers,
        )
        # GAT which will use KG
        self.use_kg = use_kg
        self.gats = None
        if use_kg:
            self.gats = utils.initialize_gnn(kg_node_dim, kg_edge_dim, n_gnn_layers)
        self.aggregator = aggr.MeanAggregation()
        # project feature space to single probability
        self.proj = nn.Linear(
            full_hidden_dim if not self.use_kg else full_hidden_dim + kg_node_dim, 1
        )
        # softmax normalizes all proj outputs to find sentence
        self.softmax = nn.Softmax(dim=-1)
        print(
            f"initialized continuityBERT with {utils.get_model_size(self)} parameters."
        )

    def forward(self, x, kgs=None):
        """
        :param x: sequence of sentence encodings from a story with shape (batch_size, seq_len, input_dim)
        :param kgs: knowledge graphs LIST, of len batch_size, each represented as the following map and shapes:
            {
                "node_feats": (n_nodes, kg_node_dim),
                "edge_indices": (2, n_edges),
                "edge_feats": (n_edges, kg_edge_dim)
            }
        :returns: sequence of logits for each sentence
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # embed input
        x = self.embedder(x)

        # obtain decider output
        x = self.decider(x)

        # if using kg, concatenate kg output to decider output
        if self.use_kg:
            x_kgs = []
            for i in range(batch_size):
                x_kg = kgs[i]["node_feats"]
                for conv in self.gats:
                    x_kg = conv(x_kg, kgs[i]["edge_indices"], kgs[i]["edge_feats"])
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


class UnresolvedBERT(nn.Module):  # UnresolvedTransformer
    """
    baseline model which finds Unresolved Storyline Errors in plots --
    i.e., whether or not the story was cut short before the storyline
    was resolved.
    """

    def __init__(
        self,
        n_heads=16,
        n_layers=6,
        n_gnn_layers=2,
        input_dim=384,
        hidden_dim=20,
        use_kg=False,
        kg_node_dim=100,
        kg_edge_dim=100,
        dropout=0.1,
    ):
        nn.Module.__init__(self)
        # embed into hidden dim
        full_hidden_dim = hidden_dim * n_heads
        self.embedder = nn.Linear(input_dim, full_hidden_dim)
        # decider decides which sentences are most important in deciding how "incomplete" story is
        self.decider = nn.Transformer(
            nhead=n_heads,
            d_model=full_hidden_dim,
            batch_first=True,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dropout=dropout,
        )
        # GAT which will use KG
        self.use_kg = use_kg
        self.gats = None
        if use_kg:
            self.gats = utils.initialize_gnn(kg_node_dim, kg_edge_dim, n_gnn_layers)
        self.aggregator = aggr.MeanAggregation()
        # project feature space to single probability
        self.proj = nn.Linear(
            full_hidden_dim if not self.use_kg else full_hidden_dim + kg_node_dim, 1
        )
        # sigmoid function to determine percentage of story cut off
        self.sigmoid = nn.Sigmoid()
        print(
            f"initialized unresolvedBERT with {utils.get_model_size(self)} parameters."
        )

    def forward(self, x, kgs=None):
        """
        :param x: sequence of sentence encodings from a story with shape (batch_size, seq_len, input_dim)
        :param kgs: knowledge graphs LIST, of len batch_size, each represented as the following map and shapes:
            {
                "node_feats": (n_nodes, kg_node_dim),
                "edge_indices": (2, n_edges),
                "edge_feats": (n_edges, kg_edge_dim)
            }
        :returns: single logit determining percentage of story that was left out
        """
        batch_size = x.shape[0]

        # embed input
        x = self.embedder(x)

        # obtain decider output
        x = self.decider(x, torch.zeros([x.shape[0], 1, x.shape[-1]]).to(device))

        # if using kg, concatenate kg output to decider output
        if self.use_kg:
            # generate gnn output for each kg
            x_kgs = []
            for i in range(batch_size):
                x_kg = kgs[i]["node_feats"]
                for conv in self.gats:
                    x_kg = conv(x_kg, kgs[i]["edge_indices"], kgs[i]["edge_feats"])
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
    x_kg = [
        {
            "node_feats": torch.rand([n_nodes, kg_node_dim]),
            "edge_indices": torch.randint(0, n_nodes, [2, n_edges]),
            "edge_feats": torch.rand([n_edges, kg_edge_dim]),
        }
    ] * batch_size

    """
    test ContinuityBERT model *without* KG
    """
    # model output
    continuity_model = ContinuityBERT()
    y_hat = continuity_model(x)
    # expected output
    y = torch.zeros((batch_size, seq_len))
    print(
        f"ContinuityBERT,noKG, output shape: {y_hat.shape}, expected shape: {y.shape}"
    )

    """
    test ContinuityBERT model *with* KG
    """
    # model output
    continuity_model = ContinuityBERT(
        use_kg=True, kg_node_dim=kg_node_dim, kg_edge_dim=kg_edge_dim
    )
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
    print(
        f"UnresolvedBERT,noKG, output shape: {y_hat.shape}, expected shape: {y.shape}"
    )

    """
    test UnresolvedBERT model *with* KG
    """
    # model output
    unresolved_model = UnresolvedBERT(
        use_kg=True, kg_node_dim=kg_node_dim, kg_edge_dim=kg_edge_dim
    )
    y_hat = unresolved_model(x, x_kg)
    # expected output
    y = torch.zeros((batch_size))
    print(f"UnresolvedBERT,KG, output shape: {y_hat.shape}, expected shape: {y.shape}")
