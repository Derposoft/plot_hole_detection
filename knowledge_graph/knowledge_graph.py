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
from corenlp import StanfordCoreNLP # we need to use our own library for server parallelism
import nltk

nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("punkt", quiet=True)
spacy_nlp = en_core_web_sm.load()
stanford_core_nlp_path="./stanford-corenlp-4.5.1"
nlp = StanfordCoreNLP(stanford_core_nlp_path, quiet=True)


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


def replace_coreferences(corefs, doc, named_entities):
    """This function is an absolute mess."""
    corefs = corefs["corefs"]

    # replace all corefs in i th coref list with this
    replace_coref_with = []
    
    # Key is sentence number; value is list of tuples of (reference_dict, coreference number)
    sentence_wise_replacements = defaultdict(list)

    sentences = nltk.sent_tokenize(doc)
    for index, coreferences in enumerate(corefs.values()):    # corefs : {[{}]} => coreferences : [{}]
        # Find which coreference to replace each coreference with. By default, replace with first reference.
        replace_with = coreferences[0]
        for reference in coreferences:      # reference : {}
            if reference["text"] in named_entities.keys() or reference["text"][reference["headIndex"]-reference["startIndex"]] in named_entities.keys():
                replace_with = reference
            sentence_wise_replacements[reference["sentNum"]-1].append((reference,index))
        replace_coref_with.append(replace_with["text"])  
    
    # sort tuples in list according to start indices for replacement 
    sentence_wise_replacements[0].sort(key=lambda tup: tup[0]["startIndex"]) 
    
    #Carry out replacement
    for index,sent in enumerate(sentences):
        # Get the replacements in ith sentence
        replacement_list = sentence_wise_replacements[index]    # replacement_list : [({},int)]
        # Replace from last to not mess up previous replacement"s indices
        for item in replacement_list[::-1]:                     # item : ({},int)
            to_replace = item[0]                                # to_replace: {}
            replace_with = replace_coref_with[item[1]]
            replaced_sent = ""
            words = nltk.word_tokenize(sent)
            
            # Add words from end till the word(s) that need(s) to be replaced
            for i in range(len(words)-1,to_replace["endIndex"]-2,-1):
                replaced_sent = words[i] + " "+ replaced_sent
            
            # Replace
            replaced_sent = replace_with + " " + replaced_sent
            # Copy starting sentence
            for i in range(to_replace["startIndex"]-2,-1,-1):
                replaced_sent = words[i] + " "+ replaced_sent
            sentences[index] = replaced_sent

    result = " ".join(sentences)
    return result


def get_coreferences(doc):
    annotated = nlp.annotate(doc, properties={
        "annotators": "coref",
        "pipelineLanguage": "en"
    })
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


def perform_ner(doc):
    """
    if args.stanford:
        named_entities = stanford_ner(doc)
        #stanford_ner.display(ner)
    if args.spacy:
        named_entities = spacy_ner(doc)
        #spacy_ner.display(named_entities)
    return named_entities
    """
    return stanford_ner(doc)

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
    pool = Pool(os.cpu_count())
    all_named_entities = pool.map(perform_ner, docs)
    
    # Save named entities for downstream processes
    for file, named_entities in zip(files, all_named_entities):
        fname = file.split("/")[-1].split(".")[0]
        if platform == "win32":
            fname = fname.split("\\")[-1]
        op_pickle_filename = ner_pickles_op + "named_entity_" + fname + ".pickle"
        with open(op_pickle_filename,"wb") as f:
            pickle.dump(named_entities, f)

    # Perform coreference resolution in parallel
    if execute_coref_resol:
        docs = generate_coreferences(docs, all_named_entities)

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

    main(args)
