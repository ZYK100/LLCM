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
