import time

import numpy as np
import pandas as pd
from tqdm import tqdm 

import wikidata as wikidata

PATH = '../data/movie-100k-wiki/'
HOP = 1
DATASET_NAME = '100k'

def get_movielens(item_file, user_file, rating_file):
    from rs_datasets import MovieLens
    ml = MovieLens(DATASET_NAME)
    item_df = ml.items
    user_df = ml.users
    rating_df = ml.ratings
    origin_len = item_df.shape[0]

    item_df = item_df.drop_duplicates(subset=['title'])
    item_df = item_df[['item_id', 'title']]
    print('Drop duplicate items: {}'.format(item_df.shape[0]-origin_len))

    rating_df = rating_df[['user_id', 'item_id', 'rating']]
    
    # save to csv
    user_df.to_csv(user_file, sep='\t', index=False)
    item_df.to_csv(item_file, sep='\t', index=False)
    rating_df.to_csv(rating_file, sep='\t', index=False)


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
    link_writer = open(link_output, "w", encoding='utf-8')
    total_item_entity = []
    notFound = []
    link_writer.write('%s\t%s\t%s\t%s\n' % ('item_id', 'item_name', 'entity_id', 'entity_name'))
    for idx, items in enumerate(tqdm(item_list)):
        entity_id, entity_name = wikidata.find_wikidata_id(items[1])

        if entity_id == "entityNotFound":
            notFound.append(items[0])
            continue
        
        link_writer.write('%s\t%s\t%s\t%s\n' % (items[0], items[1], entity_id, entity_name))
        total_item_entity.append([entity_id, items[1]])
    return total_item_entity, notFound

def cover_link_file(link_file, link_output):
    link_writer = open(link_output, "r", encoding='utf-8')
    link_writer.write('%s\t%s\n' % ('item_id', 'entity_id'))

    for line in open(link_file, encoding='utf-8').readlines()[1:]:
        item = line.strip().split('\t')[0]
        entity = line.strip().split('\t')[2]
        link_writer.write('%s\t%s\n' % (item, entity))

def process_kg(kg_output, kg_full_output, item_entity_list):
    kg_writer = open(kg_output, "w", encoding='utf-8')
    kg_full_writer = open(kg_full_output, "w", encoding='utf-8')
    kg_writer.write('%s\t%s\t%s\n' % ('head', 'relation', 'tail'))
    kg_full_writer.write('%s\t%s\t%s\t%s\t%s\t%s\n' % ('head', 'head_name', 'relation', 'relation_name', 'tail', 'tail_name'))

    triple_ctx = 0
    entities_list = [item_entity_list]

    for h in range(HOP+1):
        LIMIT = 20
        if h ==0:
            LIMIT = 500
        print('Hop: {}, length: {}'.format(h, len(entities_list)))
        temp = []  
        not_find = []
        for idx, entities in enumerate(tqdm(entities_list[-1])):
            
            json_links = wikidata.query_entity_links(entities[0], limit=LIMIT)
            related_links = wikidata.read_linked_entities(json_links)

            time.sleep(1)
            if len(related_links) ==0:
                not_find.append(entities)
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
        print(not_find)
    print('TOTAL TRIPLES: {}'.format(triple_ctx))

if __name__ == '__main__':
    user_file = PATH + 'movie.user'
    item_file = PATH + 'movie.item'
    rating_file PATH + 'movie.inter'
    link_output = PATH + 'movie.link'
    kg_output = PATH + 'movie.kg'
    full_link_output = PATH + 'movie-full.link'
    kg_full_output = PATH + 'movie-full.kg'

    '''
    # get movielens dataset
    print('Download dataset...')
    get_movielens(item_file, user_file, rating_file)
    '''
    # get all item 
    item_list = get_item_list(item_file)
    '''
    # align item to entity 
    print('align item to entity...')
    total_item_entity, notFound = process_link(full_link_output, item_list)
    print('NOT FOUND: {}'.format(len(notFound)))
    print(notFound)
    cover_link_file(full_link_output, link_output)
    '''
    total_item_entity = []
    for line in open(full_link_output, encoding='utf-8').readlines():
        item = line.strip().split('\t')[0]
        entity = line.strip().split('\t')[2]
        entity_name =  line.strip().split('\t')[3]

        total_item_entity.append([entity, entity_name])
    
    print('find relative triples...')
    process_kg(kg_output, kg_full_output, total_item_entity)
    