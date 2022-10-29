import torch
from models import bert
from data import utils
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
ospj = os.path.join


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
    encoder = utils.create_sentence_encoder()
    continuity_data = utils.encode_stories(encoder, continuity_data)
    unresolved_data = utils.encode_stories(encoder, unresolved_data)

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


def train(*, model, data, opt, criterion, epochs=10, verbose=True):
    for epoch in range(epochs):
        if verbose:
            print(f"starting epoch {epoch}")
        for i, (X, y) in enumerate(data):
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            if verbose and i%1==0:
                print(loss)


if __name__ == "__main__":
    """
    create and train baseline continuity and unresolved error models
    """
    ### hyperparameters ###
    encoder_type = "all-MiniLM-L6-v2"
    batch_size = 8
    n_heads = 16
    lr = 1e-3
    n_stories = 2 # limit # of stories for debugging purposes

    # create models
    print("creating models...")
    baseline_continuity = bert.ContinuityBERT(n_heads=n_heads, input_dim=utils.SENTENCE_ENCODER_DIM[encoder_type])
    baseline_unresolved = bert.UnresolvedBERT(n_heads=n_heads, input_dim=utils.SENTENCE_ENCODER_DIM[encoder_type])
    print("done.")

    # read data
    print("reading data...")
    continuity_dataloader, unresolved_dataloader = read_data(batch_size=batch_size, n_stories=n_stories)
    print("done.")

    # train continuity model
    print("training continuity error model...")
    opt = Adam(baseline_continuity.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train(
        model=baseline_continuity,
        data=continuity_dataloader,
        opt=opt,
        criterion=criterion,
    )
    print("done")

    # train unresolved model
    print("training unresolved error model...")
    opt = Adam(baseline_unresolved.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train(
        model=baseline_unresolved,
        data=unresolved_dataloader,
        opt=opt,
        criterion=criterion,
    )
    print("done")
