import os
import torch
import glob
import pickle
import pandas as pd
import en_core_web_sm
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import data.utils as utils

SENTENCE_TRANFORMER_MODEL = 'all-MiniLM-L6-v2'
KG_NODE_DIM = 100
KG_EDGE_DIM = utils.SENTENCE_ENCODER_DIM[SENTENCE_TRANFORMER_MODEL]


def get_node_features(adj_dict, all_spacy_entities, spacy_entities_to_index_map):
    nodes_features = list()
    visited = set()
    for entity in adj_dict:
        if entity in visited:
            continue
        else:
            visited.add(entity)

        per_node_vector = [0]*KG_NODE_DIM#len(all_spacy_entities)
        per_node_vector[spacy_entities_to_index_map[adj_dict[entity]['Type']]]=1
        nodes_features.append(per_node_vector)

        for adj_entity_and_type in adj_dict[entity]['Edges']:
            adj_entity, type_, _ = adj_entity_and_type

            if adj_entity in visited:             
                continue
            else:
                visited.add(adj_entity)

            per_node_vector = [0]*KG_NODE_DIM#len(all_spacy_entities)
            per_node_vector[spacy_entities_to_index_map[type_]]=1
            nodes_features.append(per_node_vector)
    
    return nodes_features
    
def get_edge_features(adj_dict, model):
    edge_features = list()
    for entity in adj_dict:
        for adj_entity_and_type in adj_dict[entity]['Edges']:
            _, _, relationship = adj_entity_and_type
            edge_features.append(model.encode(relationship))
    return edge_features

def create_adjacency_dict(kg_post_processed, all_spacy_entities, spacy_entities_to_index_map, model):
    adj_dict = defaultdict(dict)
    edges_list = [[],[]]
    entity_index_map = dict()
    #create adjacency dictionary                        
    for _, row in kg_post_processed.iterrows():                           
            adj_dict[row['Entity 1']]['Type']=row['Type'].strip()                               
            if 'Edges' not in adj_dict[row['Entity 1'].strip()]:
                adj_dict[row['Entity 1'].strip()]['Edges'] = [(row['Entity2'].strip(), row['Type.1'].strip(), row['Relationship'].strip())]
            else:             
                adj_dict[row['Entity 1'].strip()]['Edges'].append((row['Entity2'].strip(), row['Type.1'].strip(), row['Relationship'].strip()))

    #get all the entities including those which are classified as 'O' meaning objects
    all_entities = list(adj_dict.keys())

    
    for entity in adj_dict:
        for adj_entity_and_type in adj_dict[entity]['Edges']:
            adj_entity, type_, _ = adj_entity_and_type
            all_entities.extend([adj_entity])

    for i, entity in enumerate(all_entities):
        entity_index_map[entity]=i

    #get all the edge indices
    for entity in adj_dict:
        for adj_entity_and_type in adj_dict[entity]['Edges']:
            adj_entity, type_, _ = adj_entity_and_type
            edges_list[0].append(entity_index_map[entity])
            edges_list[1].append(entity_index_map[adj_entity])


    
    #get node features
    nodes_features = get_node_features(adj_dict, all_spacy_entities, spacy_entities_to_index_map)

    #get edge features
    edge_features = get_edge_features(adj_dict, model)

    return torch.Tensor(nodes_features), torch.Tensor(edges_list).long(), torch.Tensor(np.array(edge_features))
    

def process_csv(csv_file_name, all_spacy_entities, spacy_entities_to_index_map, model):
    kg_post_processed = pd.read_csv(csv_file_name)
    nodes_features, edges_list, edge_features = create_adjacency_dict(kg_post_processed, all_spacy_entities, spacy_entities_to_index_map, model)
    return nodes_features, edges_list, edge_features


def process_extraction_results():
    all_spacy_entities = en_core_web_sm.load().get_pipe('ner').labels
    all_spacy_entities = list(all_spacy_entities)
    all_spacy_entities.append('O')
    paths = glob.glob("./knowledge_graph/data/result/*.csv")
    spacy_entities_to_index_map = {v:k for k,v in enumerate(all_spacy_entities)}
    model = SentenceTransformer(SENTENCE_TRANFORMER_MODEL)
    kgs = {}
    for path in paths:
        node_features, edges_list, edge_features = process_csv(path, all_spacy_entities, spacy_entities_to_index_map, model)
        kgs[path.split("/")[-1].lstrip("named_entity_").split(".")[0]+".txt"] = {"node_feats": node_features, "edge_indices": edges_list, "edge_feats": edge_features}
        #print('For path:{} nodes_features:{} edges_list:{} edge_features:{}'.format(path, node_features.size(), edges_list.size(), edge_features.size()))
    return kgs


if __name__=='__main__':
    process_extraction_results()
