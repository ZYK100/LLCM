### t-SNE

Run

(1) Save the features in 'tsne.mat' by:
```
python extract.py --dataset llcm --method agw --gpu 0
```

(2) Visualizing the feature distribution with t-SNE:
```
python tsne.py --dataset llcm --method agw --gpu 0
```

### Ranking-10 list

Run

(1) Save the features in 'tsne.mat' by:
```
python extract.py --dataset llcm --method agw --gpu 0
```

(2) Visualizing the Ranking-10 list by:
```
python ranking.py
```



### Intra-Inter class distances

Run

(1) Save the features in 'tsne.mat' by:
```
python extract.py --dataset llcm --method agw --gpu 0
```

(2) Visualizing the intra-inter class distances by:
```
python intra_inter-distance.py
```
