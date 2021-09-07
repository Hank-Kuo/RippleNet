# RippleNet
This repository is a **PyTorch** implementation of RippleNet ([arXiv](https://arxiv.org/abs/1803.03467)):
> RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems  
Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, Minyi Guo  
The 27th ACM International Conference on Information and Knowledge Management (CIKM 2018)

For the authors' official TensorFlow implementation, see [hwwang55/RippleNet](https://github.com/hwwang55/RippleNet).

RippleNet is a deep end-to-end model that naturally incorporates the knowledge graph into recommender systems.
Ripple Network overcomes the limitations of existing embedding-based and path-based KG-aware recommendation methods by introducing preference propagation, which automatically propagates users' potential preferences and explores their hierarchical interests in the KG.

## Usage
**train model**
```bash 
python preprocess_ml-100k.py # ouput: ratings_final.txt, kg_final.txt
python train.py
```

**Eval model**
```bash
python evaluate.py
```

## Requirment

## Dataset
### Movielens 100k
link: https://drive.google.com/drive/folders/1OkDVEqetvOrtbuWebxl4y1JlZ_YjjfWj

- **Dataset source**: Movielens 100k
- **Knoweldge graph**: freebase subgraph

- *.inter file: 

| user_id | item_id  | rating | timestamp |
| :-----: | :------: | :----: | :-------: |
|  196    |   242    |   3    | 881250949 |
|  186    |   302    |   3    | 891717742 |
|   22    |   377    |   1    | 878887116 |
|  244    |   51     |   2    | 880606923 |


- *.link

|item_id | entity_id|
|--|:-----:|
| 476 | m.08gjyx|
| 362 | m.035_kw|
| 1612 | m.0fwd14|
| 1181 | m.0gksh95|

- *.item

| item_id | movie_title | release_year | class |
|:---:|:---|:---:|:-----------------------------|
| 1 | Toy Story | 1995 | Animation Children's Comedy |
| 2 | GoldenEye | 1995 | Action Adventure Thriller |
| 3 | Four Rooms | 1995 | Thriller |
| 4 | Get Shorty | 1995 | Action Comedy Drama |

- *.user

| user_id | age | gender | occupation | zip_code |
|:---:|:---|:---:|:--------| :------: |
| 1	| 24 | M | technician | 85711 |
| 2 | 53 | F | other | 94043 |
| 3 | 23 | M | writer | 32067 | 
| 4 | 24 | M | technician | 43537 |


- *.kg

| head_id | relation_id | tail_id|
|:---:|:---------|---:|
| m.04ctbw8 | film.producer.film | m.0bln8|
| m.0c3wmn | film.film.actor | m.02vxxgs|
| m.04t36 | film.film_genre...| m.05sbv3|
| m.08jl3y | film.film.actor	| m.0v187kf|

- Detail 

|  | Movielens 10k |
|:----------:|:-------|
| #Users | 944 | 
| #Items | 1683 |
| #Interactions | 100,000 |
| #Triplets | 91631 |
| #Entities | 34713 |
| #Relations | 26 |
| Rating | [1-5] |


### Movielens 1M
- **[link](https://github.com/hwwang55/RippleNet)**

- **Dataset source**: Movielens 1M
- **Knoweldge graph**: Mircosoft Satori


- Detail 

|  | Movielens 1M |
|:----------:|:-------|
| #Users | 6,040 | 
| #Items | 3,629 |
| #Interactions | 836,478 |
| #Triplets | 91631 |
| #Entities | 34713 |
| #Relations | 26 |
| Rating | [1-5] |


## Result

**MovieLens 1M**:
|  | Train AUC | Train ACC | Eval AUC | Eval ACC | Test AUC | Test ACC |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Epoch 5 | 0.9533 | 0.8836 | 0.9216 | 0.8466 | 0.9206 | 0.8457 |

**MovieLens 100k**
|  | Train AUC | Train ACC | Eval AUC | Eval ACC | Test AUC | Test ACC |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Epoch 6 | 0.9604 |  0.8931 | 0.8997 | 0.8248 | 0.9006 | 0.8250 |


## paper dataset 
number of users: 6036
number of items: 2445
number of entities (containing items): 182011
number of relations: 12

### Required packages
The code has been tested running under Python 3.6, with the following packages installed (along with their dependencies):
- tqdm
- logging
- pytorch >= 1.0
- numpy >= 1.14.5
- sklearn >= 0.19.1


