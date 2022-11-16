from copy import deepcopy
import nltk
import numpy as np
import os
from pathlib import Path
import shutil
from sys import platform
from typing import List, Tuple
#from knowledge_graph import gnn_data_utils as kg_utils
from knowledge_graph.gnn_data_utils import process_extraction_results

nltk.download("averaged_perceptron_tagger")
ROOT = Path(__file__)
osl = os.listdir
ospj = os.path.join


def get_datafiles() -> list:
    """
    returns list of stories(.txt file) in raw_story_file folder
    """
    return [x for x in Path(ROOT.parent / "raw").iterdir() if str(x).endswith(".txt")]


def negater(sentence: str) -> list:
    """
    Basic logic
    1. Check if the word is a verb using nltk tagger
    2. If the word is verb, look for antonyms
    3. If antonym exist get a random antonym and put it in the word's place
    4. If antonym does not exist, put "not" in front of the word if that's not a corpula, if the word is a corpula put "not" after the word.
    """
    wordnet = nltk.corpus.wordnet
    to_be_verbs = {"was", "is", "are", "am"}
    tgt = sentence.split(" ")
    tags = nltk.pos_tag(tgt)
    res = list()
    negated = False
    for word, tag in zip(tgt, tags):
        if tag[1][0] == "V" and not negated:
            negated = True
            # is the verb a to-be verb?
            if word in to_be_verbs:
                res.append(f"{word} not")
                continue
            
            # if not a to-be verb, can we find an antonym?
            antonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    cands = l.antonyms()
                    if cands:
                        antonyms.append(cands[0].name())
            antonyms = list(set(antonyms))
            if len(antonyms) > 0: res.append(np.random.choice(antonyms, 1, replace=False)[0])

            # no antonym exists; just prepend "not" and hope things work out
            else: res.append(f"not {word}")
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
    samples = np.random.choice(range(len(sentences)), min(n, len(sentences)), replace=False)
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
    most_sentences_to_remove = max(n+1, int(p * n_sentences))
    
    # Given number of lines will be random #See below 0 to 20% of Number of Sentences
    samples = np.random.choice(range(1, most_sentences_to_remove), min(n, most_sentences_to_remove-1), replace=False)

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


def generate_synthetic_data(n=10, train_ratio=0.8):
    dataset = get_datafiles()
    n_docs = len(dataset)
    for doc_idx in range(len(dataset)):
        train_test_prefix = "train/" if doc_idx < n_docs*train_ratio else "test/"
        doc_path = dataset[doc_idx]
        with open(doc_path, "r", encoding="utf8") as document_f:
            document = " ".join([x.strip() for x in document_f.readlines()])
            X_continuity, y_continuity = generate_continuity_errors(document, n)
            X_unresolved, y_unresolved = generate_unresolvedstory_errors(document, n)
            for i in range(n):
                if i >= len(X_continuity) or i >= len(X_unresolved): break
                doc_name = str(doc_path).split("\\" if platform=="win32" else "/")[-1].split(".")[0]
                continuity_path = ROOT.parent / f"synthetic/{train_test_prefix}synthetic_{doc_name}_continuity{i}.txt"
                unresolved_path = ROOT.parent / f"synthetic/{train_test_prefix}synthetic_{doc_name}_unresolved{i}.txt"
                X, y = X_continuity[i], y_continuity[i]
                write_synthetic_datapoint_to_file(
                    X=X, y=y, path=continuity_path, plot_hole_type="continuity"
                )
                X, y = X_unresolved[i], y_unresolved[i]
                write_synthetic_datapoint_to_file(
                    X=X, y=y, path=unresolved_path, plot_hole_type="unresolved"
                )


def generate_kgs(data_files_path):
    # 1. copy all data_files to knowledge_graph/data/input/
    kg_path = "./knowledge_graph"
    for data_file in osl(data_files_path):
        if not data_file.endswith(".txt"): continue
        shutil.copy(ospj(data_files_path, data_file), ospj(f"{kg_path}/data/input/", data_file))
        break

    # 2. run commands to create knowledge graph outputs
    os.chdir(kg_path)
    os.system(f"python3 knowledge_graph.py stanford")
    os.system(f"python3 relation_extractor.py")
    os.system(f"python3 create_structured_csv.py")
    os.chdir("..")

    # 3. run knowledge graph tensor generation code
    kgs = process_extraction_results()

    # 4. clean up knowledge_graph/data folders
    folders_to_cleanup = [
        f"{kg_path}/data/input/",
        f"{kg_path}/data/output/kg/",
        f"{kg_path}/data/output/ner",
        f"{kg_path}/data/result/"
    ]
    for folder in folders_to_cleanup:
        for file in osl(folder):
            if file != ".gitignore":
                os.remove(ospj(folder, file))    

    # 5. return generated knowledge graphs
    return kgs


if __name__ == "__main__":
    generate_synthetic_data()
    generate_kgs("data/synthetic/test")
