import argparse
from data import utils
from models import bert
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
import torch
import torch.nn as nn
from torch.optim import Adam
device = "cuda" if torch.cuda.is_available() else "cpu"
PR_THRESHOLD = None


def test(*, model, test_data, metrics="f1", verbose=True):
    """
    :param model: the model to test
    :param test_data: test dataloader
    :param metrics: one of either "f1" or "mse".
    :param verbose: whether or not to print extra output
    :returns: nothing. prints metrics
    """
    # collect metrics
    y_preds = []
    y_true = []
    for _, (X, y) in enumerate(test_data):
        with torch.no_grad():
            y_preds.append(model(X))
        y_true.append(y)
    y_preds, y_true = y_preds, y_true

    # calculate metrics
    results = None
    y_true = torch.cat(y_true).cpu().flatten()
    y_preds = torch.cat(y_preds).cpu().flatten()
    if metrics == "f1":
        y_preds = y_preds >= PR_THRESHOLD
        results = f1_score(y_true, y_preds)
    elif metrics == "mse":
        results = mean_squared_error(y_true, y_preds)
    else:
        print(f"{metrics} metric not implemented. please choose one of [f1, mse].")
    print(f"{metrics} score: {results}")


def train(*, model, train_data, test_data, opt, criterion, epochs=10, metrics="f1", verbose=True):
    """
    :param model: the model to test
    :param train_data: train dataloader
    :param test_data: test dataloader
    :param verbose: whether or not to print extra output
    :returns: nothing. trains given model using train_data and tests it every epoch with test_data
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
    parser.add_argument("--n_stories", default=2, type=int, help="number of synthetic datapoints to use")
    parser.add_argument("--train_ratio", default=0.8, type=float, help="train ratio")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--n_heads", default=16, type=int)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--pr_threshold", default=0.5, type=float)
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
    n_stories = config.n_stories
    n_epochs = config.n_epochs
    train_ratio = config.train_ratio
    PR_THRESHOLD = config.pr_threshold

    # create models
    print("creating models...")
    baseline_continuity = bert.ContinuityBERT(n_heads=n_heads, input_dim=utils.SENTENCE_ENCODER_DIM[encoder_type])
    baseline_unresolved = bert.UnresolvedBERT(n_heads=n_heads, input_dim=utils.SENTENCE_ENCODER_DIM[encoder_type])
    baseline_continuity.to(device)
    baseline_unresolved.to(device)
    print("done.")

    # read data
    print("reading data...")
    continuity_train, unresolved_train = utils.read_data(
        batch_size=batch_size,
        n_stories=int(n_stories*train_ratio),
        data_path="data/synthetic/train",
        cache_path="data/encoded/train",
    )
    continuity_test, unresolved_test = utils.read_data(
        batch_size=batch_size,
        n_stories=int(n_stories*(1-train_ratio)),
        data_path="data/synthetic/test",
        cache_path="data/encoded/test",
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
        epochs=n_epochs,
        metrics="f1",
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
        epochs=n_epochs,
        metrics="mse",
    )
    print("done")
