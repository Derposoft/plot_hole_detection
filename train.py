import argparse
from models import bert
from data import utils
import torch.nn as nn
from torch.optim import Adam


def test(*, model, test_data, metrics="f1", verbose=True):
    """
    :param model: the model to test
    :param test_data: test dataloader
    :param metrics: one of either "f1" or "mse".
    :param verbose: whether or not to print extra output
    """
    y_preds = []
    y_true = []
    for _, (X, y) in enumerate(test_data):
        y_preds.append(model(X))
        y_true.append(y)
    # TODO calculate metrics here using y_preds and y_true. find f1 score if metrics="f1", otherwise calculate mse.
    results = "TODO"
    # TODO print metrics here
    print(f"{metrics} score: {results}")


def train(*, model, train_data, test_data, opt, criterion, epochs=10, metrics="f1", verbose=True):
    """
    :param model: the model to test
    :param train_data: train dataloader
    :param test_data: test dataloader
    :param verbose: whether or not to print extra output
    """
    for epoch in range(epochs):
        if verbose:
            print(f"starting epoch {epoch}")
        for i, (X, y) in enumerate(train_data):
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            if verbose and i%1==0:
                print(f"batch {i} loss: {loss.item()}")
        test(model=model, test_data=test_data, metrics=metrics, verbose=verbose)


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
    continuity_train, unresolved_train = utils.read_data(
        batch_size=batch_size, n_stories=n_stories, data_path="data/synthetic/train", cache_path="data/encoded/train"
    )
    continuity_test, unresolved_test = utils.read_data(
        batch_size=batch_size, n_stories=n_stories, data_path="data/synthetic/test", cache_path="data/encoded/test"
    )
    print("done.")

    # train continuity model
    print("training continuity error model...")
    opt = Adam(baseline_continuity.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train(
        model=baseline_continuity,
        train_data=continuity_train,
        test_data=continuity_test,
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
        train_data=unresolved_train,
        test_data=unresolved_test,
        opt=opt,
        criterion=criterion,
    )
    print("done")
