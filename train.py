import argparse
from models import bert
from data import utils
import torch.nn as nn
from torch.optim import Adam


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nstories", default=2, type=int, help="number of synthetic datapoints to use")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--n_heads", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--encoder_type", default="all-MiniLM-L6-v2", type=str,
        choices=["all-MiniLM-L6-v2", "paraphrase-albert-small-v2"])
    config = parser.parse_args()
    return config


if __name__ == "__main__":
    """
    create and train baseline continuity and unresolved error models
    """
    ### hyperparameters ###
    config = parse_args()
    encoder_type = config.encoder_type
    batch_size = config.batch_size
    n_heads = config.n_heads
    lr = config.lr
    n_stories = config.nstories

    # create models
    print("creating models...")
    baseline_continuity = bert.ContinuityBERT(n_heads=n_heads, input_dim=utils.SENTENCE_ENCODER_DIM[encoder_type])
    baseline_unresolved = bert.UnresolvedBERT(n_heads=n_heads, input_dim=utils.SENTENCE_ENCODER_DIM[encoder_type])
    print("done.")

    # read data
    print("reading data...")
    continuity_dataloader, unresolved_dataloader = utils.read_data(batch_size=batch_size, n_stories=n_stories)
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
