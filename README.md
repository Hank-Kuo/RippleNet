# RippleNet

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

## kk
Dataset	#User	#Item	#Interaction	Sparsity	#Entity	#Relation	#Triple
ml-1m	6,040	3,629	836,478	0.9618	79,388	51	385,923