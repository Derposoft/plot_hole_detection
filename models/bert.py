"""
file containing the definitions for the baseline BERT models
that to find each of the 2 kinds of plot holes.
"""
from typing import List
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class ContinuityBERT(nn.Module):
    """
    baseline model which finds Continuity Errors in plots --
    i.e., which sentences are plot holes and which ones are not.
    """
    def __init__(self, n_heads=16, encoder_name="all-MiniLM-L6-v2"):
        nn.Module.__init__(self)
        # encoder encodes input sentences
        self.encoder, self.encoder_dim = create_sentence_encoder(encoder_name)
        # decider decides which sentences are continuity errors
        self.decider = nn.Transformer(nhead=n_heads, d_model=self.encoder_dim, batch_first=True)
        # project feature space to single probability
        self.proj = nn.Linear(self.encoder_dim, 1)
        # softmax normalizes all proj outputs to find sentence
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: List[List[str]]):
        """
        :param x: sequence of sentences from a story with shape batch_size, n_sentences, 
        :returns: sequence of logits for each sentence
        """
        x = torch.stack([torch.Tensor(self.encoder.encode(_x)) for _x in x])
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
    def __init__(self, n_heads=16, encoder_name="all-MiniLM-L6-v2"):
        nn.Module.__init__(self)
        # encoder encodes input sentences
        self.encoder, self.encoder_dim = create_sentence_encoder(encoder_name)
        # decider decides which sentences are most important in deciding how "incomplete" story is
        self.decider = nn.Transformer(nhead=n_heads, d_model=self.encoder_dim, batch_first=True)
        # project feature space to single probability
        self.proj = nn.Linear(self.encoder_dim, 1)
        # sigmoid function to determine percentage of story cut off
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.stack([torch.Tensor(self.encoder.encode(_x)) for _x in x])
        x = self.decider(x, torch.zeros([x.shape[0], 1, x.shape[-1]]))
        x = self.proj(x)
        x = x.reshape([x.shape[0], -1])
        return self.sigmoid(x)


def create_sentence_encoder(encoder_name="all-MiniLM-L6-v2"):
    """
    :param encoder_name: the name of the pretrained encoder 
        model to use. currently supported are "all-MiniLM-L6-v2" and 
        "paraphrase-albert-small-v2".
    :returns: (encoder_model, encoder_output_dim) tuple.
    """
    SENTENCE_ENCODER_DIM = {
        "all-MiniLM-L6-v2": 384,
        "paraphrase-albert-small-v2": 768
    }
    assert encoder_name in SENTENCE_ENCODER_DIM, f"encoder name must be one of {list(SENTENCE_ENCODER_DIM.keys())}"
    encoder = SentenceTransformer(f"sentence-transformers/{encoder_name}")
    encoder.eval()
    return encoder, SENTENCE_ENCODER_DIM[encoder_name]


if __name__ == "__main__":
    batch_size = 2
    
    """
    test ContinuityBERT model
    """
    with open("data/synthetic/synthetic_text1_continuity0.txt", "r") as f:
        lines = f.readlines()
    # model output
    continuity_model = ContinuityBERT()
    x = [lines[1:]] * batch_size
    y_hat = continuity_model(x)
    # expected output
    tgt_idx = int(lines[0].split()[1])
    y = torch.zeros((batch_size, len(lines)-1))
    y[:,tgt_idx] = 1
    print(f"output shape: {y_hat.shape}, expected shape: {y.shape}")

    """
    test UnresolvedBERT model
    """
    with open("data/synthetic/synthetic_text1_unresolved0.txt", "r") as f:
        lines = f.readlines()
    # model output
    unresolved_model = UnresolvedBERT()
    x = [lines[1:]] * batch_size
    y_hat = unresolved_model(x)
    # expected output
    y = torch.zeros((batch_size, 1))
    print(f"output shape: {y_hat.shape}, expected shape: {y.shape}")
