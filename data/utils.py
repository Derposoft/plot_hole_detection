from typing import List
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
ospj = os.path.join


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


class StoryDataset(Dataset):
    def __init__(self, X, y):
        Dataset.__init__(self)
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def read_data(batch_size=8, data_path="data/synthetic", encoder="all-MiniLM-L6-v2", n_stories=None):
    """
    :param batch_size: batch_size for output dataloaders
    :param data_path: location of data
    :returns: tuple of (continuity_dataloader, unresolved_dataloader) dataloaders

    first parses data files at data_path, then preprocesses them by encoding each story
    by sentence by pading smaller stories with 0s at the end so that all stories are the 
    same length. labels for continuity errors are 1-hot encoded. returns dataloaders of these stories.
    """
    # parse all data files in data_path and separate them by error type
    data_files = os.listdir(data_path)
    data_files = [x for x in data_files if x.endswith(".txt")]
    continuity_data = []
    continuity_labels = []
    unresolved_data = []
    unresolved_labels = []
    for data_file in data_files:
        with open(ospj(data_path, data_file), "r") as f:
            lines = f.readlines()
            problem, label = lines[0].split()
            if problem == "continuity":
                continuity_data.append(lines[1:])
                continuity_labels.append(int(label))
            elif problem == "unresolved":
                unresolved_data.append(lines[1:])
                unresolved_labels.append(float(label))
    if n_stories:
        continuity_data = continuity_data[:min(len(continuity_data), n_stories)]
        continuity_labels = continuity_labels[:min(len(continuity_labels), n_stories)]
        unresolved_data = unresolved_data[:min(len(unresolved_data), n_stories)]
        unresolved_labels = unresolved_labels[:min(len(unresolved_labels), n_stories)]
    
    # encode all data file sentences using encoder
    encoder = create_sentence_encoder()
    continuity_data = encode_stories(encoder, continuity_data)
    unresolved_data = encode_stories(encoder, unresolved_data)

    # pad all stories to meet the length of the longest story
    longest_story_length = max([len(story) for story in continuity_data])
    continuity_data = [
        F.pad(story, (0, 0, 0, longest_story_length - len(story)))
        for story in continuity_data
    ]
    unresolved_data = [
        F.pad(story, (0, 0, 0, longest_story_length - len(story)))
        for story in unresolved_data
    ]
    continuity_data = torch.stack(continuity_data)
    unresolved_data = torch.stack(unresolved_data)

    # 1-hot encode continuity error labels, turn labels into tensors
    continuity_labels = torch.eye(longest_story_length)[continuity_labels]
    unresolved_labels = torch.FloatTensor(unresolved_labels)

    # return dataloaders for each error type
    continuity_dataloader = DataLoader(StoryDataset(continuity_data, continuity_labels), batch_size=batch_size)
    unresolved_dataloader = DataLoader(StoryDataset(unresolved_data, unresolved_labels), batch_size=batch_size)
    return continuity_dataloader, unresolved_dataloader
