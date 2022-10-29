"""
file containing the definitions for the baseline BERT models
that to find each of the 2 kinds of plot holes.
"""
from typing import List
import torch
import torch.nn as nn


class ContinuityBERT(nn.Module):
    """
    baseline model which finds Continuity Errors in plots --
    i.e., which sentences are plot holes and which ones are not.
    """
    def __init__(self, n_heads=16, input_dim=384):
        nn.Module.__init__(self)
        # decider decides which sentences are continuity errors
        self.decider = nn.Transformer(nhead=n_heads, d_model=input_dim, batch_first=True)
        # project feature space to single probability
        self.proj = nn.Linear(input_dim, 1)
        # softmax normalizes all proj outputs to find sentence
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: sequence of sentence encodings from a story with shape (batch_size, seq_len, input_dim) 
        :returns: sequence of logits for each sentence
        """
        x = self.decider(x, torch.zeros(x.shape))
        x = self.proj(x)
        x = x.reshape([x.shape[0], -1])
        return self.softmax(x)


class UnresolvedBERT(nn.Module):
    """
    baseline model which finds Unresolved Storyline Errors in plots --
    i.e., whether or not the story was cut short before the storyline 
    was resolved.
    """
    def __init__(self, n_heads=16, input_dim=384):
        nn.Module.__init__(self)
        # decider decides which sentences are most important in deciding how "incomplete" story is
        self.decider = nn.Transformer(nhead=n_heads, d_model=input_dim, batch_first=True)
        # project feature space to single probability
        self.proj = nn.Linear(input_dim, 1)
        # sigmoid function to determine percentage of story cut off
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        :param x: sequence of sentence encodings from a story with shape (batch_size, seq_len, input_dim) 
        :returns: single logit determining percentage of story that was left out
        """
        x = self.decider(x, torch.zeros([x.shape[0], 1, x.shape[-1]]))
        x = self.proj(x)
        x = x.reshape([x.shape[0]])
        return self.sigmoid(x)


if __name__ == "__main__":
    batch_size = 2
    seq_len = 100
    
    """
    test ContinuityBERT model
    """
    # model output
    continuity_model = ContinuityBERT()
    x = torch.rand([batch_size, seq_len, 384])
    y_hat = continuity_model(x)
    # expected output
    y = torch.zeros((batch_size, seq_len))
    print(f"output shape: {y_hat.shape}, expected shape: {y.shape}")

    """
    test UnresolvedBERT model
    """
    # model output
    unresolved_model = UnresolvedBERT()
    x = torch.rand([batch_size, seq_len, 384])
    y_hat = unresolved_model(x)
    # expected output
    y = torch.zeros((batch_size))
    print(f"output shape: {y_hat.shape}, expected shape: {y.shape}")
