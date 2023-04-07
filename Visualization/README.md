### 1、t-SNE

(1) Save the features in 'tsne.mat' by:
```
python extract.py --dataset llcm --method agw --gpu 0
```

(2) Visualizing the feature distribution with t-SNE:
```
python tsne.py --dataset llcm --method agw --gpu 0
```
![tsne](https://github.com/ZYK100/LLCM/blob/main/Visualization/imgs/tsne_0.jpg)


### 2、Intra-Inter class distances

(1) Save the features in 'tsne.mat' by:
```
python extract.py --dataset llcm --method agw --gpu 0
```

(2) Visualizing the intra-inter class distances by:
```
python intra_inter-distance.py
```
![intra_inter-distance](https://github.com/ZYK100/LLCM/blob/main/Visualization/imgs/intra_inter.jpg)

### 3、Ranking-10 list

(1) Save the features in 'tsne.mat' by:
```
python extract.py --dataset llcm --method agw --gpu 0
```

(2) Visualizing the Ranking-10 list by:
```
python ranking.py
```
![Ranking-10](https://github.com/ZYK100/LLCM/blob/main/Visualization/imgs/ranking.jpg)
