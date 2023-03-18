import argparse
import en_core_web_sm
import glob
import json
from multiprocessing import Pool
import nltk
import numpy as np
import os
import pickle
import sys
import torch
import traceback

from knowledge_graph.corenlp import StanfordCoreNLP
from models.model_utils import SENTENCE_ENCODER_DIM
from sentence_transformers import SentenceTransformer


nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("punkt", quiet=True)
spacy_nlp = en_core_web_sm.load()
stanford_core_nlp_path="./stanford-corenlp-4.5.1"
nlp = None


os.environ["TOKENIZERS_PARALLELISM"] = "false"
CAP_TOT_EDGES = 100
SENTENCE_TRANFORMER_MODEL = 'all-MiniLM-L6-v2'
KG_NODE_DIM = 100
KG_EDGE_DIM = SENTENCE_ENCODER_DIM[SENTENCE_TRANFORMER_MODEL]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


def perform_triple_extraction_pipeline(doc):
    annotated = nlp.annotate(doc, properties={
        "annotators": "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,openie",
        "pipelineLanguage": "en"
    })
    return json.loads(annotated)


def make_kg(doc_pipeline_output):
    # Graph object representing {u: {v1: rel1, v2: rel2, ...}}
    node2idx = {}
    edge_list = []
    edge_feat = []
    for sentence in doc_pipeline_output["sentences"]:
        if CAP_TOT_EDGES > 0 and len(edge_list) > CAP_TOT_EDGES:
            break
        for triple in sentence["openie"]:
            # Extract subject, relation, and object from knowledge triple and add to g
            s, r, o = triple["subject"], triple["relation"], triple["object"]
            if o not in node2idx:
                node2idx[o] = len(node2idx)
            if s not in node2idx:
                node2idx[s] = len(node2idx)
            edge_list.append([node2idx[s], node2idx[o]])
            edge_feat.append(r)
    
    # Encode node_feats, edge_list, edge_feats in required format for PyG
    node_feat = torch.eye(KG_NODE_DIM)[np.array(list(range(len(node2idx)))) % KG_NODE_DIM]
    edge_list = torch.Tensor(edge_list).t().contiguous().long()
    edge_feat = model.encode(edge_feat, convert_to_tensor=True)
    return {
        "node_feats": node_feat,
        "edge_indices": edge_list,
        "edge_feats": edge_feat,
    }


def start_pipeline():
    global nlp
    global model
    if not nlp:
        nlp = StanfordCoreNLP(stanford_core_nlp_path, quiet=True, threads=1, timeout=60000)
    if not model:
        model = SentenceTransformer(SENTENCE_TRANFORMER_MODEL).to(device)


def stop_pipeline():
    global nlp
    if nlp:
        nlp.close()


def generate_kgs(docs):
    try:
        start_pipeline()
        # Create KGs in parallel
        pool = Pool(os.cpu_count())
        all_triples_info = pool.map(perform_triple_extraction_pipeline, docs)
        return pool.map(make_kg, all_triples_info)
    except:
        stop_pipeline()
        traceback.print_exc()
        sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Usage: python3 knowledge_graph.py path/to/input/data"
    )
    parser.add_argument(
        "input_dir",
        type=str,
    )
    args = parser.parse_args()

    # Get documents and run kg generator
    files = glob.glob(os.path.join(args.input_dir,  "*.txt"))
    docs = []
    for file in files:
        with open(file,"r") as f:
            lines = f.read().splitlines()
        doc = " ".join(lines)
        docs.append(doc)
    kgs = generate_kgs(docs)
    
    with open("knowledge_graphs.pkl", "wb") as f:
        pickle.dump(kgs, f)
