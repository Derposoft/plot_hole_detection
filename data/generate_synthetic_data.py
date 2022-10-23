from copy import deepcopy
import nltk
import numpy as np
from pathlib import Path
from sys import platform
from typing import List, Tuple

nltk.download("averaged_perceptron_tagger")
ROOT = Path(__file__)


def is_txt(path) -> bool:
    return str(path).split(".")[-1] == "txt"


def get_datafiles() -> list:
    """
    returns list of stories(.txt file) in raw_story_file folder
    """
    return [x for x in Path(ROOT.parent / "raw").iterdir() if is_txt(x)]

def negater(sentence: str) -> list:
    """
    Basic logic
    1. Check if the word is a verb using nltk tagger
    2. If the word is verb, look for antonyms
    3. If antonym exist get a random antonym and put it in the word's place
    4. If antonym does not exist, put "not" in front of the word if that's not a corpula, if the word is a corpula put "not" after the word.
    """
    wordnet = nltk.corpus.wordnet
    corpula = {"was", "is", "are", "am"}
    
    tgt = sentence.split(" ")
    tags = nltk.pos_tag(tgt) 
    res = list()
    negated = False
    for word, tag in zip(tgt, tags):
        if tag[1][0] == "V" and not negated:
            negated = True
            # word is verb; check for antonyms (change verb to VBD, VBN if past tense)
            antonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    cands = l.antonyms()
                    if cands:
                        antonyms.append(cands[0].name())
            antonyms = list(set(antonyms))
            # antonym exists; use random antonym
            if len(antonyms) > 0: res.append(np.random.choice(antonyms, 1, replace=False)[0])
            # no antonym exists; use not
            else: res.append(f"{word} not" if word in corpula else f"not {word}")
        else: res.append(word)
    return " ".join(res)

def generate_continuity_errors(document: str, n: int) -> Tuple[List[str], List[int]]:
    """
    negate random lines in a story to create continuity errors storyliens
    :param document: string document.
    :param n: number of samples to generate.
    :returns: (X, y) tuple for X=list of synthetic documents, y=list of labels
    """
    sentences =  [x.strip() for x in document.split(".") if x != ""]
    samples = np.random.choice(range(len(sentences)), n, replace=False)
    X = []
    for sample in samples:
        X.append(deepcopy(sentences))
        X[-1][sample] = negater(X[-1][sample])
    X = [".\n".join(x) for x in X]
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
    sentences = [x.strip() for x in document.split(".") if x != ""]
    n_sentences = len(sentences)
    
    # Given number of lines will be random #See below 0 to 20% of Number of Sentences
    samples = np.random.choice(range(1, max(n+1, int(p * n_sentences))), n, replace=False)

    # Create n text with n lines from the last removed   
    for sample in samples:
        X.append(".\n".join(sentences[:-sample]))
    y = samples / n_sentences
    return X, y


def write_synthetic_datapoint_to_file(X, y, path, plot_hole_type):
    """
    write a synthetic datapoint to a file.
    :param X: synthetic document
    :param y: synthetic label
    :param path: path to write the file to
    :param plot_hole_type: type of plot hole, will be written at top of document
    :returns: None. file will be written at path. first line will be "plot_hole_type y", and
    rest of the lines will be X.
    """
    with open(path, "w", encoding="utf-8") as synthetic_document_f:
        synthetic_document_f.write(f"{plot_hole_type} {y}\n")
        synthetic_document_f.write(X[1:])


if __name__ == "__main__":
    n = 10
    dataset = get_datafiles()
    for doc_path in dataset:
        with open(doc_path, "r", encoding="utf8") as document_f:
            document = " ".join([x.strip() for x in document_f.readlines()])
            X_continuity, y_continuity = generate_continuity_errors(document, n)
            X_unresolved, y_unresolved = generate_unresolvedstory_errors(document, n)
            for i in range(n):
                doc_name = str(doc_path).split("\\" if platform=="win32" else "/")[-1].split(".")[0]
                continuity_path = f"data/synthetic/synthetic_{doc_name}_continuity{i}.txt"
                unresolved_path = f"data/synthetic/synthetic_{doc_name}_unresolved{i}.txt"
                X, y = X_continuity[i], y_continuity[i]
                write_synthetic_datapoint_to_file(
                    X=X, y=y, path=continuity_path, plot_hole_type="continuity"
                )
                X, y = X_unresolved[i], y_unresolved[i]
                write_synthetic_datapoint_to_file(
                    X=X, y=y, path=unresolved_path, plot_hole_type="unresolved"
                )
