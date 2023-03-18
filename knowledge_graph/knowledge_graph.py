import argparse
from collections import defaultdict
import glob
import json
from multiprocessing import Pool
import os
import pickle
from sys import platform

#from stanfordcorenlp import StanfordCoreNLP
import en_core_web_sm
from knowledge_graph.corenlp import StanfordCoreNLP # we need to use our own library for server parallelism
import nltk

nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("punkt", quiet=True)
spacy_nlp = en_core_web_sm.load()
stanford_core_nlp_path="./stanford-corenlp-4.5.1"
nlp = StanfordCoreNLP(stanford_core_nlp_path, quiet=True, threads=1)


def stanford_ner(doc, stanford_ner_path="./stanford-ner-2018-10-16"):
    # Load stanford NER Tagger
    stanford_ner_tagger = nltk.tag.StanfordNERTagger(
        stanford_ner_path+"/classifiers/english.all.3class.distsim.crf.ser.gz",
        stanford_ner_path+"/stanford-ner.jar",
    )

    # Perform NER
    sentences = nltk.sent_tokenize(doc)
    ner = []
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        tagged = stanford_ner_tagger.tag(words)
        ner.append(tagged)
    ner_dict = {}
    for tup in ner[0]:
        ner_dict[tup[0]] = tup[1]
    return ner_dict


def spacy_ner(doc):
    doc = spacy_nlp(doc)
    ner = [(X.text, X.label_) for X in doc.ents]
    ner_dict = {}
    for tup in ner:
        ner_dict[tup[0]] = tup[1]
    return ner_dict


def replace_coreferences(corefs: dict, doc, named_entities):
    """This function is an absolute mess."""
    corefs = corefs["corefs"]

    sentenceidx2corefs = defaultdict(list)
    sentences = nltk.sent_tokenize(doc)
    for index, coreferences in enumerate(corefs.values()):
        for reference in coreferences:
            sentenceidx2corefs[reference["sentNum"]-1].append(reference)
    
    #print(sentenceidx2corefs[0], "SENTENCE REPLACEMENTS")
    
    #Carry out replacement
    for index, sentence in enumerate(sentences):
        print("Sentence that we're changing: ", sentence)
        words = nltk.word_tokenize(sentence)
        for coref in sentenceidx2corefs[index]:
            words[coref["startIndex"]] = coref["text"]
            for i in range(coref["startIndex"]+1, coref["endIndex"]):
                words[i] = ""
        words = [word for word in words if word != ""]
        new_sentence = " ".join(words)
        sentences[index] = new_sentence
        print("New sentence:", new_sentence)

    result = " ".join(sentences)
    return result


def get_coreferences(doc):
    annotated = nlp.annotate(doc, properties={
        #"annotators": "coref",
        #"annotators": "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,openie",
        "annotators": "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,openie",
        "pipelineLanguage": "en"
    })
    with open("corefs.json", "w") as f:
        f.write(annotated)
    return json.loads(annotated)


def generate_coreferences(docs, all_named_entities):
    """
    :param docs: gets coreferences for all of the given docs
    """
    # Get coreferences for each of the docs in parallel
    pool = Pool(os.cpu_count())
    all_coreferences = pool.map(get_coreferences, docs)
    nlp.close()

    # Apply coreferences for each of the docs
    processed_docs = pool.starmap(replace_coreferences, zip(all_coreferences, docs, all_named_entities))
    return processed_docs


STANFORD_NER = False
SPACY_NER = False
def perform_ner(doc):
    if STANFORD_NER:
        named_entities = stanford_ner(doc)
    elif SPACY_NER:
        named_entities = spacy_ner(doc)
    else:
        named_entities = spacy_ner(doc)
    return named_entities


def main(args):
    verbose = args.verbose
    execute_coref_resol = args.optimized
    output_path = "./data/output/"
    ner_pickles_op = output_path + "ner/"
    coref_cache_path = output_path + "caches/"
    coref_resolved_op = output_path + "kg/"
    
    # Collect docs
    files = glob.glob("./data/input/*.txt")
    docs = []
    for file in files:
        with open(file,"r") as f:
            lines = f.read().splitlines()
        doc = " ".join(lines)
        docs.append(doc)
    
    # Perform NER in parallel
    print("[kg] Starting NER")
    pool = Pool(os.cpu_count())
    all_named_entities = pool.map(perform_ner, docs)
    print("[kg] NER complete")
    
    # Save named entities for downstream processes
    for file, named_entities in zip(files, all_named_entities):
        fname = file.split("/")[-1].split(".")[0]
        if platform == "win32":
            fname = fname.split("\\")[-1]
        op_pickle_filename = ner_pickles_op + "named_entity_" + fname + ".pickle"
        with open(op_pickle_filename,"wb") as f:
            pickle.dump(named_entities, f)

    # Perform coreference resolution in parallel
    print("[kg] Starting coref resolution")
    if execute_coref_resol:
        docs = generate_coreferences(docs, all_named_entities)
    print("[kg] corefs done")

    for file, doc in zip(files, docs):
        op_filename = (
            coref_resolved_op + file.split("/")[-1].split("\\")[-1]
            if platform == "win32"
            else coref_resolved_op + file.split("/")[-1]
        )
        with open(op_filename,"w+") as f:
            f.write(doc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Usage: python3 knowledge_graph.py <nltk/stanford/spacy> [optimized,verbose,nltk,stanford,spacy]"
    )
    parser.add_argument("--nltk", action="store_true")
    parser.add_argument("--stanford", action="store_true")
    parser.add_argument("--spacy", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--optimized", action="store_true")
    args = parser.parse_args()

    if args.stanford:
        STANFORD_NER = True
    if args.spacy:
        SPACY_NER = True

    main(args)
