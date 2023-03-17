import argparse
import glob
import json
from multiprocessing import Pool
import os
import pickle
import torch
import traceback

import en_core_web_sm
from knowledge_graph.corenlp import StanfordCoreNLP # we need to use our own library for server parallelism
import nltk

from sentence_transformers import SentenceTransformer
import data.utils as utils


nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("punkt", quiet=True)
spacy_nlp = en_core_web_sm.load()
stanford_core_nlp_path="./stanford-corenlp-4.5.1"
nlp = StanfordCoreNLP(stanford_core_nlp_path, quiet=True, threads=1, timeout=60000)


SENTENCE_TRANFORMER_MODEL = 'all-MiniLM-L6-v2'
KG_NODE_DIM = 100
KG_EDGE_DIM = utils.SENTENCE_ENCODER_DIM[SENTENCE_TRANFORMER_MODEL]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(SENTENCE_TRANFORMER_MODEL).to(device)


def make_kg(doc_pipeline_output):
    # Graph object representing {u: {v1: rel1, v2: rel2, ...}}
    print("[kg] parsing openie output")
    node2idx = {}
    edge_list = []
    edge_feat = []
    for sentence in doc_pipeline_output["sentences"]:
        for triple in sentence["openie"]:
            # Extract subject, relation, and object from knowledge triple and add to g
            s, r, o = triple["subject"], triple["relation"], triple["object"]
            if o not in node2idx:
                node2idx[o] = len(node2idx)
            if s not in node2idx:
                node2idx[s] = len(node2idx)
            edge_list.append([node2idx[s], node2idx[o]])
            edge_feat.append(r)
    print("[kg] done parsing openie output")
    
    # Encode node_feats, edge_list, edge_feats in required format for PyG
    node_feat = torch.eye(KG_NODE_DIM)[list(range(len(node2idx)))]
    edge_list = torch.Tensor(edge_list).t().contiguous()
    edge_feat = model.encode(edge_feat, convert_to_tensor=True)
    return node_feat, edge_list, edge_feat


def perform_triple_extraction_pipeline(doc):
    annotated = nlp.annotate(doc, properties={
        "annotators": "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,openie",
        "pipelineLanguage": "en"
    })
    return json.loads(annotated)


def generate_kgs(docs):
    try:
        # Create KGs in parallel
        print("[kg] performing triple extraction")
        pool = Pool(os.cpu_count())
        all_triples_info = pool.map(perform_triple_extraction_pipeline, docs)
        nlp.close()
        print("[kg] triple extraction complete. building kgs")
        node_feats, edge_lists, edge_feats = list(zip(pool.map(make_kg, all_triples_info)))
        print("[kg] kg creation done")
        return node_feats, edge_lists, edge_feats
    except:
        nlp.close()
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Usage: python3 knowledge_graph.py path/to/input/data"
    )
    parser.add_argument(
        "input_dir",
        type=str,
    )
    args = parser.parse_args()
    node_feats, edge_list, edge_feats = generate_kgs(args.input_dir)
    print("[kg] writing to pkl...")
    with open("knowledge_graphs.pkl", "wb") as f:
        pickle.dump((node_feats, edge_list, edge_feats), f)
    print("[kg] done!")
