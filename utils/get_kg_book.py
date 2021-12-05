import time

import numpy as np
import pandas as pd
from tqdm import tqdm 

import wikidata1 as wikidata

PATH = '../data/book-wiki/'
HOP = 3

def get_book(item_file, user_file, rating_file):
    from rs_datasets import BookCrossing
    bx = BookCrossing()
    rating_df = bx.ratings
    user_df = bx.users
    item_df = bx.items

    item_df = item_df.drop_duplicates(subset=['title'])
    item_df = item_df[['item_id', 'title']]
  
    user_df.to_csv(user_file, sep='\t', index=False)
    rating_df.to_csv(rating_file, sep='\t', index=False)
    item_df.to_csv(item_file, sep='\t', index=False)

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
        time.sleep(0.5)
        entity_id, entity_name, isCheck = wikidata.find_wikidata_id(items[1], category=['novel', 'book', 'write', 'wrote','written','author', 'fiction', 'story', 'stories'], limit=3)

        if entity_id == "entityNotFound":
            notFound.append(items[0])
            continue
        
        link_writer.write('%s\t%s\t%s\t%s\t%s\n' % (items[0], items[1], entity_id, entity_name, str(isCheck)))
        total_item_entity.append([entity_id, items[1]])
    return total_item_entity, notFound

def cover_link_file(link_file, link_output):
    link_writer = open(link_output, "w", encoding='utf-8')
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
    triple_ctx_list = [0, triple_ctx]
    entities_list = [item_entity_list]

    for h in range(HOP):
        LIMIT = 5
    
        if h ==0:
            LIMIT = 50
        print('Hop: {}, length: {}, triple:{}'.format(h, len(entities_list), triple_ctx_list[-2] -triple_ctx_list[-1] ))
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
        triple_ctx_list.append(triple_ctx)
        if h==0:
            print(not_find)
    print('TOTAL TRIPLES: {}'.format(triple_ctx))

if __name__ == '__main__':
    user_file = PATH + 'book.user'
    item_file = PATH + 'book.item'
    rating_file= PATH + 'book.inter'
    link_output = PATH + 'book.link'
    kg_output = PATH + 'book.kg'
    full_link_output = PATH + 'book-full.link'
    kg_full_output = PATH + 'book-full.kg'

    '''
    # get book dataset
    print('Download dataset...')
    get_book(item_file, user_file, rating_file)
    '''
    
    # get all item 
    item_list = get_item_list(item_file)

    # align item to entity 
    print('align item to entity...')
    total_item_entity, notFound = process_link(full_link_output, item_list)
    print('NOT FOUND: {}'.format(len(notFound)))
    print(notFound)
    '''
    print('Covering full-link to link file')
    cover_link_file(full_link_output, link_output)
    
    total_item_entity = []
    for line in open(full_link_output, encoding='utf-8').readlines():
        item = line.strip().split('\t')[0]
        entity = line.strip().split('\t')[2]
        entity_name =  line.strip().split('\t')[3]

        total_item_entity.append([entity, entity_name])
    
    print('Find relative triples...')
    process_kg(kg_output, kg_full_output, total_item_entity)
    '''
    