import os
import subprocess
import glob
import pandas as pd
from multiprocessing import Pool


def process_large_corpus(file):
    p = subprocess.Popen(["./process_large_corpus.sh", file, file + "-out.csv"], stdout=subprocess.PIPE)
    output, err = p.communicate()
    return (output, err)


def Stanford_Relation_Extractor():
    files = glob.glob(os.getcwd() + "/data/output/kg/*.txt")
    current_directory = os.getcwd()
    os.chdir(current_directory + "/stanford-openie")
    pool = Pool(os.cpu_count())
    outputs_and_errors = pool.map(process_large_corpus, files)


if __name__ == '__main__':
    Stanford_Relation_Extractor()
