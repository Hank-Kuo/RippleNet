import numpy as np

PATH = '../data/movie/'
SEP = '::'
THRESHOLD = 4 

def convert_rating(rating_file, output_file, item_index_old2new):
    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()
    for line in open(rating_file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP)

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= THRESHOLD:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    
    writer = open(output_file, 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg(kg_files, output_file, entity_id2index, relation_id2index):
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 1

    writer = open(output_file, 'w', encoding='utf-8')

    for file in kg_files:
        for line in file:
            array = line.strip().split('\t')
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]

            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(555)
    
    align_file = PATH + 'item_index2entity_id_rehashed.txt'
    rating_file = PATH +  'ratings.dat'


    output_file = PATH+ 'ratings_final.txt'
    kg_output_file = PATH+ 'kg_final.txt'

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    entity_cnt = 1
    for line in open(align_file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = entity_cnt
        entity_id2index[satori_id] = entity_cnt
        entity_cnt += 1

    convert_rating(rating_file, output_file, item_index_old2new)
    
    kg_files = []
    kg_files.append(open(PATH+'kg_part1_rehashed.txt', encoding='utf-8'))
    kg_files.append(open(PATH+'kg_part2_rehashed.txt', encoding='utf-8'))
    convert_kg(kg_files, kg_output_file, entity_id2index, relation_id2index)
