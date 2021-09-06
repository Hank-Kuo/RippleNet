# RippleNet
This repository is a **PyTorch** implementation of RippleNet ([arXiv](https://arxiv.org/abs/1803.03467)):
> RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems  
Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, Minyi Guo  
The 27th ACM International Conference on Information and Knowledge Management (CIKM 2018)

For the authors' official TensorFlow implementation, see [hwwang55/RippleNet](https://github.com/hwwang55/RippleNet).

RippleNet is a deep end-to-end model that naturally incorporates the knowledge graph into recommender systems.
Ripple Network overcomes the limitations of existing embedding-based and path-based KG-aware recommendation methods by introducing preference propagation, which automatically propagates users' potential preferences and explores their hierarchical interests in the KG.
## Test

## Usage
- train model
```python 
python train.py
```

- Eval model
```python
python evaluate.py
```

## Requirment

## Dataset
link: https://drive.google.com/drive/folders/1OkDVEqetvOrtbuWebxl4y1JlZ_YjjfWj
**Dataset source**: Movielens 100k
**Knoweldge graph**: freebase subgraph

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

|  | Movielens |
|:----------:|:-------|
| #Users | 944 | 
| #Items | 1683 |
| #Interactions | 100,000 |
| #Triplets | 91631 |
| #Entities | 34713 |
| #Relations | 26 |
Rating
[1-5]





## Paper
number of users: 6036
number of items: 2445
number of entities (containing items): 182011
number of relations: 12

use paper code to run our dataset 
epoch 0    train auc: 0.8950  acc: 0.8118    eval auc: 0.8756  acc: 0.7997    test auc: 0.8760  acc: 0.7952
epoch 1    train auc: 0.9074  acc: 0.8232    eval auc: 0.8825  acc: 0.8024    test auc: 0.8832  acc: 0.8022
epoch 2    train auc: 0.9258  acc: 0.8452    eval auc: 0.8935  acc: 0.8151    test auc: 0.8970  acc: 0.8198
epoch 3    train auc: 0.9311  acc: 0.8523    eval auc: 0.8941  acc: 0.8175    test auc: 0.8964  acc: 0.8172
epoch 4    train auc: 0.9415  acc: 0.8638    eval auc: 0.8973  acc: 0.8214    test auc: 0.8996  acc: 0.8232
epoch 5    train auc: 0.9502  acc: 0.8784    eval auc: 0.8991  acc: 0.8257    test auc: 0.8993  acc: 0.8227
epoch 6    train auc: 0.9597  acc: 0.8939    eval auc: 0.8972  acc: 0.8243    test auc: 0.8998  acc: 0.8252
epoch 7    train auc: 0.9694  acc: 0.9083    eval auc: 0.8963  acc: 0.8219    test auc: 0.9013  acc: 0.8255
epoch 8    train auc: 0.9766  acc: 0.9214    eval auc: 0.8920  acc: 0.8189    test auc: 0.8978  acc: 0.8245
epoch 9    train auc: 0.9806  acc: 0.9299    eval auc: 0.8887  acc: 0.8156    test auc: 0.8926  acc: 0.8208

## kk
Dataset	#User	#Item	#Interaction	Sparsity	#Entity	#Relation	#Triple
ml-1m	6,040	3,629	836,478	0.9618	79,388	51	385,923


### Required packages
The code has been tested running under Python 3.6, with the following packages installed (along with their dependencies):
- pytorch >= 1.0
- numpy >= 1.14.5
- sklearn >= 0.19.1
