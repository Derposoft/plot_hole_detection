from pathlib import Path
from typing import List, Tuple
import numpy as np
from copy import deepcopy
#from negation_conversion import applyNegation


def is_txt(path) -> bool:
    return str(path).split(".")[-1] == "txt"


def get_datafiles() -> list:
    """
    Returns list of stories(.txt file) in raw_story_file folder
    """    
    ROOT = Path(__file__)
    return [x for x in Path(ROOT.parent / "raw").iterdir() if is_txt(x)]


def generate_continuity_errors(document: str, n: int) -> Tuple[List[str], List[int]]:
    """
    negate random lines in a story to create continuity errors storyliens
    :param document: string document.
    :param n: number of samples to generate.
    :returns: (X, y) tuple for X=list of synthetic documents, y=list of labels
    """
    sentences = document.split(".")
    samples = np.random.choice(range(len(sentences)), n, replace=False)
    X = []
    negate = {"was", "is", "are", "am"}
    for sample in samples:
        X.append(deepcopy(sentences))
        #X[-1][sample] = "".join([word if word not in negate or not word.endswith("ed") else word + " not " for word in X[-1][sample]])
        synthetic_document = []
        for word in X[-1][sample].split(" "):
            if word.endswith("ed"):
                synthetic_document.append("was not "+word)
            elif word in negate:
                synthetic_document.append(word + " not")
            else:
                synthetic_document.append(word)
        X[-1][sample] = " ".join(synthetic_document)
    y = samples
    return X, y


def generate_unresolvedstory_errors(document: str, n: int, p:float=0.1) -> Tuple[List[str], List[int]]:
    """
    removes random n lines from the end of a story to create unresolved storyliens
    :param document: string document.
    :param n: number of samples to generate.
    :param p: percentage of sentences to cut off of the end at most
    """ 
    X = []
    # Preprocessing - remove new line character and empty lines
    sentences = document.split(".")
    n_sentences = len(sentences)
    
    # Given number of lines will be random #See below 0 to 20% of Number of Sentences
    samples = np.random.choice(range(1, max(n+1, int(p * n_sentences))), n, replace=False)

    # Create n text with n lines from the last removed   
    for sample in samples:
        X.append(" ".join(sentences[:-sample]))
    y = samples / n_sentences
    return X, y


if __name__ == "__main__":
    n = 10
    dataset = get_datafiles()
    for document in dataset:
        with open(document, "r", encoding="utf8") as document:
            document = " ".join([x.strip() for x in document.readlines()])
            X_continuity, y_continuity = generate_continuity_errors(document, n)
            X_unresolved, y_unresolved = generate_unresolvedstory_errors(document, n)
        print(X_continuity[0][y_continuity[0]], y_continuity[0])
        print(X_unresolved[0], y_unresolved[0])
