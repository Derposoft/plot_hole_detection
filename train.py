import argparse
from data import utils
import json
from models import bert
import numpy as np
import random
from scipy.stats import ttest_1samp
from sklearn.metrics import f1_score, mean_squared_error
import torch
import torch.nn as nn
from torch.optim import Adam
import knowledge_graph.gnn_data_utils as kgutils
from time import time
device = "cuda" if torch.cuda.is_available() else "cpu"
PR_THRESHOLD = None


def set_seed(seed):
    """
    :param seed: seed to use for reproducibility purposes
    :returns: None. sets seed as per https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def test(*, model, test_data, metrics="f1", verbosity=10):
    """
    :param model: the model to test
    :param test_data: test dataloader
    :param metrics: one of either "f1" or "mse".
    :param verbosity: whether or not to print extra output, lower=more verbose
    :returns: nothing. prints metrics
    """
    # collect metrics
    y_preds = []
    y_true = []
    for _, (X, y, kgs) in enumerate(test_data):
        X, y = X.to(device), y.to(device)
        for kg in kgs:
            for k in kg: kg[k] = kg[k].to(device)
        with torch.no_grad():
            y_preds.append(model(X, kgs))
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
    if verbosity > 0:
        print(f"{metrics} score: {results}")
    return results


def train(*, model, train_data, test_data, opt, criterion, epochs=10, metrics="f1", verbosity=5):
    """
    :param model: the model to test
    :param train_data: train dataloader
    :param test_data: test dataloader
    :param verbosity: whether or not to print extra output, lower=more verbose
    :returns: nothing. trains given model using train_data and tests it every epoch with test_data
    """
    best_metric = 0 if metrics == "f1" else float("inf")
    for epoch in range(epochs):
        start_time = time()
        tot_loss = 0
        for i, (X, y, kgs) in enumerate(train_data):
            X, y = X.to(device), y.to(device)
            for kg in kgs:
                for k in kg: kg[k] = kg[k].to(device)
            y_hat = model(X, kgs)
            loss = criterion(y_hat, y)
            tot_loss += loss.item()
            loss.backward()
            opt.step()
        tot_loss /= len(train_data)
        results = None
        if (epoch+1) % verbosity == 0:
            results = test(model=model, test_data=test_data, metrics=metrics, verbosity=0)
            if metrics == "f1": best_metric = max(best_metric, results)
            else: best_metric = min(best_metric, results)
        if verbosity <= 0 or (epoch+1) % verbosity == 0:
            results_str = f", test {metrics}: {results:0.5}" if results != None else ""
            print(
                f"epoch {epoch+1} time: {time()-start_time:0.3}s, train loss: {tot_loss:0.4}{results_str}"
            )
    print(f"post-training summary -- best {metrics}: {best_metric}")
    return best_metric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_stories", default=1000, type=int, help="number of stories to use (for both test and train)")
    parser.add_argument("--n_synth", default=1, type=int, help="number of synthetic datapoints to use per story")
    parser.add_argument("--train_ratio", default=0.5, type=float, help="train ratio")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--n_heads", default=8, type=int)
    parser.add_argument("--n_layers", default=3, type=int)
    parser.add_argument("--n_gnn_layers", default=2, type=int)
    parser.add_argument("--hidden_dim", default=20, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--n_runs", default=5, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--pr_threshold", default=0.3, type=float)
    parser.add_argument("--encoder_type", default="all-MiniLM-L6-v2", type=str,
        choices=["all-MiniLM-L6-v2", "paraphrase-albert-small-v2"])
    parser.add_argument("--model_type", default="continuity_bert", type=str,
        choices=["continuity_bert", "unresolved_bert", "continuity_bert_kg", "unresolved_bert_kg"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--verbosity", default=1, type=int, help="verbosity of output if != 0; lower is more verbose")
    parser.add_argument("--settings_json", default="", type=str, help="JSON with optimal settings for the given model")
    config = parser.parse_args()
    config = vars(config)
    settings_json = config.get("settings_json", "")
    if settings_json != "":
        with open(settings_json, "r") as f:
            user_provided_settings = json.load(f).get(config["model_type"], {})
        config.update(user_provided_settings)
    return config


if __name__ == "__main__":
    """
    create and train baseline continuity and unresolved error models
    """
    ### hyperparameters ###
    config = parse_args()
    set_seed(config["seed"])
    model_type = config["model_type"]
    train_ratio = config["train_ratio"]
    PR_THRESHOLD = config["pr_threshold"]

    # read data
    batch_size = config["batch_size"]
    n_stories = config["n_stories"]
    n_synth = config["n_synth"]
    use_kg = "kg" in model_type
    encoder_type = config["encoder_type"]
    print("reading data...")
    continuity_train, unresolved_train = utils.read_data(
        batch_size=batch_size,
        n_stories=n_stories,
        n_synth=n_synth,
        data_path="data/synthetic/train",
        cache_path="data/encoded/train",
        get_kgs=use_kg,
        encoder=encoder_type,
    )
    continuity_test, unresolved_test = utils.read_data(
        batch_size=batch_size,
        n_stories=n_stories,
        n_synth=n_synth,
        data_path="data/synthetic/test",
        cache_path="data/encoded/test",
        get_kgs=use_kg,
        encoder=encoder_type,
    )
    print("done.")

    # create training artifacts
    print("creating training artifacts...")
    if "continuity" in model_type:
        model_class = bert.ContinuityBERT
        train_data, test_data = continuity_train, continuity_test
        criterion = nn.CrossEntropyLoss()
        metrics = "f1"
    else:
        model_class = bert.UnresolvedBERT
        train_data, test_data = unresolved_train, unresolved_test
        criterion = nn.MSELoss()
        metrics = "mse"
    print("done.")
    
    # start runs
    print(f"training {model_type} model...")
    best_test_metrics = []
    for i in range(config["n_runs"]):
        print(f"run {i+1} start -- seed={config['seed']}")
        # create model
        model = model_class(
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            n_gnn_layers=config["n_gnn_layers"],
            hidden_dim=config["hidden_dim"],
            input_dim=utils.SENTENCE_ENCODER_DIM[encoder_type],
            use_kg=use_kg,
            kg_node_dim=kgutils.KG_NODE_DIM,
            kg_edge_dim=kgutils.KG_EDGE_DIM,
            dropout=config["dropout"],
        )
        model = model.to(device)
        opt = Adam(model.parameters(), lr=config["lr"])
        # train model
        best_test_metric = train(
            model=model,
            train_data=train_data,
            test_data=test_data,
            opt=opt,
            criterion=criterion,
            epochs=config["n_epochs"],
            metrics=metrics,
            verbosity=config["verbosity"],
        )
        best_test_metrics.append(best_test_metric)
        config["seed"] += 1
    for i in range(len(best_test_metrics)):
        print(f"run {i+1}: {best_test_metrics[i]}")
    print(f"done.")

    # calculate final metrics
    UNRESOLVED_ERROR_HUMAN_BENCHMARK = 2.51e-3
    UNRESOLVED_ERROR_RANDOM_MODEL = 1.37e-2
    CONTINUITY_ERROR_HUMAN_BENCHMARK = 0.5
    CONTINUITY_ERROR_RANDOM_MODEL = 0.026
    confidence_interval_95_zval = 1.96
    if "unresolved" in model_type:
        t_human, p_human = ttest_1samp(
            best_test_metrics, UNRESOLVED_ERROR_HUMAN_BENCHMARK, alternative="less"
        )
        t_random, p_random = ttest_1samp(
            best_test_metrics, UNRESOLVED_ERROR_RANDOM_MODEL, alternative="less"
        )
    else:
        t_human, p_human = ttest_1samp(
            best_test_metrics, CONTINUITY_ERROR_HUMAN_BENCHMARK, alternative="less"
        )
        t_random, p_random = ttest_1samp(
            best_test_metrics, CONTINUITY_ERROR_RANDOM_MODEL, alternative="less"
        )
    print(f"t,p-val for human<model: {t_human},{p_human}, significant: {p_human<0.05}")
    print(f"t,p-val for random<model: {t_random},{p_random}, significant: {p_random<0.05}")
    std_dev = np.std(best_test_metrics)
    mean = np.mean(best_test_metrics)
    print(f"95% CI: {mean}+/-{std_dev*confidence_interval_95_zval}")
