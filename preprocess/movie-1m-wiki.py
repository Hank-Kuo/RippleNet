import argparse
import numpy as np

'''
input: *.user, *.inter, *.item, *.link, *.kg
ouptut: output.txt, kg_final.txt, user2idx.txt, entity2id.txt, relation2idx.txt, item2idx.txt
    rating_final.txt: user_id item_id class
    kg_final.txt: h r t
'''

PATH = '../data/movie-1m-wiki/'
THRESHOLD = 4 

def count_line(file, text):
    with open(file, 'r', encoding='utf-8') as fp:
        size = len(fp.readlines())
        print(text+': {}'.format(size-1))

def save_mapping(output_file, dic):
    writer = open(output_file, 'w', encoding='utf-8')
    for k, v in dic.items():
        writer.write('%s\t%d\n' % (k, v))
    writer.close()

if __name__ == '__main__':
    np.random.seed(555)

    item2idx = dict()
    user2idx = dict()
    entity2idx = dict() 
    relation2idx = dict()
    user_pos_ratings = dict()
    user_neg_ratings = dict()
    

    user_file = PATH+ 'movie.user'
    item_file = PATH+ 'movie.item'
    link_file = PATH+ 'movie.link'
    inter_file = PATH+ 'movie.inter'
    kg_file = PATH+ 'movie.kg'

    output_file = PATH+ 'ratings_final.txt'
    kg_output_file = PATH+ 'kg_final.txt'
    user2idx_file = PATH+ 'user2idx.txt'
    entity2idx_file = PATH+ 'entity2idx.txt'
    relation2idx_file = PATH+ 'relation2idx.txt'
    item2idx_file = PATH+ 'item2idx.txt'

    ### ------------------------------------------- ###

    count_line(user_file, 'Total number of user')
    count_line(item_file, 'Total number of item')
    count_line(link_file, 'Total number of link')
    count_line(inter_file, 'Total number of inter')
    count_line(kg_file, 'Total number of kg')

    ### ------------------------------------------- ###
    # deal with link file for align entity with item
    i = 1
    for line in open(link_file, encoding='utf-8').readlines()[1:]:
        item = line.strip().split('\t')[0]
        entity = line.strip().split('\t')[1]
        item2idx[item] = i
        entity2idx[entity] = i
        i += 1
    
    ### ------------------------------------------- ###
    # deal with inter file
    item_set = set(item2idx.values())
    inter_cnt = 0 

    for line in open(inter_file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split('\t')

        item = array[1]
        # item 不在 link 中踢除
        if item not in item2idx:
            continue
        item_index = item2idx[item]
        user = int(array[0])
        rating = float(array[2])
        inter_cnt += 1

        # 區分成 0 or 1 
        if rating >= THRESHOLD:
            if user not in user_pos_ratings:
                user_pos_ratings[user] = set()
            user_pos_ratings[user].add(item_index)
        else:
            if user not in user_neg_ratings:
                user_neg_ratings[user] = set()
            user_neg_ratings[user].add(item_index)
    
    print('-------------------------------------')
    print('converting inter file ...')
    print('Number of inter: {}'.format(inter_cnt))

    writer = open(output_file, 'w', encoding='utf-8')
    user_cnt = 0
    
    for user, pos_item_set in user_pos_ratings.items():
        if user not in user2idx:
            user2idx[user] = user_cnt
            user_cnt += 1
        user_index = user2idx[user]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))

        unwatched_set = item_set - pos_item_set
        if user in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('Number of users: %d' % user_cnt)
    print('Number of items: %d' % len(item_set))

    ### ------------------------------------------- ###
    print('-------------------------------------')
    print('converting kg file ...')
    entity_cnt = len(entity2idx)
    relation_cnt = 1

    writer = open(kg_output_file, 'w', encoding='utf-8')

    for line in open(kg_file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split('\t')
        head = array[0]
        relation = array[1]
        tail = array[2]

        if head not in entity2idx:
            entity2idx[head] = entity_cnt
            entity_cnt += 1
        head_id = entity2idx[head]

        if tail not in entity2idx:
            entity2idx[tail] = entity_cnt
            entity_cnt += 1
        tail_id = entity2idx[tail]

        if relation not in relation2idx:
            relation2idx[relation] = relation_cnt
            relation_cnt += 1
        relation_id = relation2idx[relation]

        writer.write('%d\t%d\t%d\n' % (head_id, relation_id, tail_id))
    writer.close()
    print('Number of entities (containing items): %d' % entity_cnt)
    print('Number of relations: %d' % relation_cnt)

    print('-------------------------------------')
    print('save mapping file ...')
    save_mapping(entity2idx_file, entity2idx)
    save_mapping(relation2idx_file, relation2idx)
    save_mapping(item2idx_file, item2idx)
    save_mapping(user2idx_file, user2idx)
