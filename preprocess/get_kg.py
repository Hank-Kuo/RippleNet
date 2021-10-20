import argparse
import time

import numpy as np
import pandas as pd
from tqdm import tqdm 

import utils.wikidata as wikidata

PATH = '../data/ml-100k/'
HOP = 2 

def get_item_list(file):
    """
    return list([id, name])
    """
    item_list = []
    for line in open(file, encoding='utf-8').readlines()[1:]:
        item_id = line.strip().split('\t')[0]
        item_name = line.strip().split('\t')[1]
        item_list.append([item_id, item_name])
    return item_list

def process_link(link_output, item_list):
    """
    return list([entity_id, entity_name])
    """
    link_writer = open(link_output, "w")
    total_item_entity = []
    notFound = []

    for idx, items in enumerate(tqdm(item_list)):
        entity_id = wikidata.find_wikidata_id(items[1])

        if entity_id == "entityNotFound":
            notFound.append(items[0])
            continue
        
        link_writer.write('%s\t%s\n' % (items[0], entity_id))
        total_item_entity.append([entity_id, items[1]])
    return total_item_entity, notFound

def process_kg(kg_output, kg_full_output, item_entity_list):
    kg_writer = open(kg_output, "w")
    kg_full_writer = open(kg_full_output, "w")

    triple_ctx = 0
    entities_list = [item_entity_list]

    for h in range(HOP+1):
        print('hop: {}'.format(h))
        temp = []   

        for idx, entities in enumerate(tqdm(entities_list[-1])):
            json_links = wikidata.query_entity_links(entities[0])
            related_links = wikidata.read_linked_entities(json_links)
            time.sleep(1)
            for related_entity, related_name, relation, relation_name in related_links:
                result = dict(
                    head=entities[0],
                    head_name=entities[1],
                    relation = relation,
                    relation_name= relation_name,
                    tail=related_entity,
                    tail_name=related_name,
                )
                temp.append([related_entity, related_name])
                triple_ctx +=1 
                kg_writer.write('%s\t%s\t%s\n' % (entities[0], relation, related_entity))
                kg_full_writer.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (entities[0], entities[1], relation, relation_name, related_entity, related_name))
            
        entities_list.append(temp) 

    print('TOTAL TRIPLES: {}'.format(triple_ctx))

if __name__ == '__main__':
    item_file = PATH + 'ml-100k.item'
    link_output = PATH + 'ml-100k.link'
    kg_output = PATH + 'ml-100k.kg'
    kg_full_output = PATH + 'ml-100k-full.kg'

    # get all item 
    item_list = get_item_list(item_file)
    
    # align item to entity 
    print('align item to entity...')
    total_item_entity, notFound = process_link(link_output, item_list)
    print('NOT FOUND: {}'.format(len(notFound)))
    print(notFound)

    print('find relative triples...')
    process_kg(kg_output, kg_full_output, total_item_entity)
    '''
    results_list = []
    notFound = []
    link_writer = open(link_output, "w")
    kg_writer = open(kg_output, "w")

    for idx, items in enumerate(item_list):
        if idx % 100==0:
            print('propcess: {}'.format(idx))
    
        entity_id = wikidata.find_wikidata_id(items[1])
        link_writer.write('%s\t%s\t\n' % (items[0], entity_id))
        
        if entity_id == "entityNotFound":
            notFound.append(items[0])
            continue

        json_links = wikidata.query_entity_links(entity_id)
        related_links = wikidata.read_linked_entities(json_links)
        
        # 1 hop
        for related_entity, related_name, relation, relation_name in related_links:
            result = dict(
                head=entity_id,
                head_name=items[1],
                relation = relation,
                relation_name= relation_name,
                tail=related_entity,
                tail_name=related_name,
            )
            results_list.append(result)
            kg_writer.write('%s\t%s\t%s\t\n' % (entity_id,relation , related_entity))
        
            # 2 hop
            json_links_2 = wikidata.query_entity_links(related_entity)
            related_links_2 = wikidata.read_linked_entities(json_links_2)
            for related_entity_2, related_name_2, relation_2, relation_name_2 in related_links_2:
                result = dict(
                    head=related_entity,
                    head_name=related_name,
                    relation = relation_2,
                    relation_name= relation_name_2,
                    tail=related_entity_2,
                    tail_name=related_name_2,
                )
                results_list.append(result)
                kg_writer.write('%s\t%s\t%s\t\n' % (related_entity,relation_2 , related_entity_2))

                # 3 hop
                json_links_3 = wikidata.query_entity_links(related_entity_2)
                related_links_3 = wikidata.read_linked_entities(json_links_3)
                for related_entity_3, related_name_3, relation_3, relation_name_3 in related_links_3:
                    result = dict(
                        head=related_entity_2,
                        head_name=related_name_2,
                        relation = relation_3,
                        relation_name= relation_name_3,
                        tail=related_entity_3,
                        tail_name=related_name_3,
                    )
                    results_list.append(result)
                    kg_writer.write('%s\t%s\t%s\t\n' % (related_entity_2,relation_3 , related_entity_3))
    
    link_writer.close()

    results_list = pd.DataFrame(results_list)
    results_list.to_csv(kg_full_output, sep="\t", index=False)
    print('NOT FOUND ITEM: {}'.format(len(notFound)))
    print(notFound)
    print('TOTAL TRIPLES: {}'.format(results_list.size()))

    '''