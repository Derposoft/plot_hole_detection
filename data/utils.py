from typing import List
from sentence_transformers import SentenceTransformer
import torch


SENTENCE_ENCODER_DIM = {
    "all-MiniLM-L6-v2": 384,
    "paraphrase-albert-small-v2": 768
}


def create_sentence_encoder(encoder_name="all-MiniLM-L6-v2"):
    """
    :param encoder_name: the name of the pretrained encoder 
        model to use. currently supported are "all-MiniLM-L6-v2" and 
        "paraphrase-albert-small-v2".
    :returns: (encoder_model, encoder_output_dim) tuple.
    """
    assert encoder_name in SENTENCE_ENCODER_DIM, f"encoder name must be one of {list(SENTENCE_ENCODER_DIM.keys())}"
    encoder = SentenceTransformer(f"sentence-transformers/{encoder_name}")
    encoder.eval()
    return encoder


def encode_stories(encoder, stories: List[List[str]]):
    output = []
    for story in stories:
        output.append(torch.stack([torch.Tensor(encoder.encode(sentence)) for sentence in story]))
    return output
