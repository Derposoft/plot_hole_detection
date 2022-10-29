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
