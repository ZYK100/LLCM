### Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification
Authors: [Yukang Zhang](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=Ma51U80AAAAJ), [Hanzi Wang*](https://scholar.google.com/citations?user=AmJaPdUAAAAJ&hl=zh-CN&oi=sra)

[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Diverse_Embedding_Expansion_Network_and_Low-Light_Cross-Modality_Benchmark_for_Visible-Infrared_CVPR_2023_paper.pdf) (CVPR 2023).

### Abstract

For the visible-infrared person re-identification (VIReID) task, one of the major challenges is the modality gaps between visible (VIS) and infrared (IR) images. However, the training samples are usually limited, while the modality gaps are too large, which leads that the existing methods cannot effectively mine diverse cross-modality clues. To handle this limitation, we propose a novel augmentation network in the embedding space, called diverse embedding expansion network (DEEN). The proposed DEEN can effectively generate diverse embeddings to learn the informative feature representations and reduce the modality discrepancy between the VIS and IR images. Moreover, the VIReID model may be seriously affected by drastic illumination changes, while all the existing VIReID datasets are captured under sufficient illumination without significant light changes. Thus, we provide a low-light cross-modality (LLCM) dataset, which  contains 46,767 bounding boxes of 1,064 identities captured by 9 RGB/IR cameras. Extensive experiments on the SYSU-MM01, RegDB and LLCM datasets show the superiority of the proposed DEEN over several other state-of-the-art methods. 
![image](https://github.com/ZYK100/LLCM/blob/main/imgs/img1.png)

### Dataset download
Please send a signed [dataset release agreement](https://github.com/ZYK100/LLCM/blob/main/LLCM%20Dataset%20Agreement/LLCM%20DATASET%20RELEASE%20AGREEMENT.pdf) copy to zhangyk@stu.xmu.edu.cn. If your application is passed, we will send the download link of the dataset.

![image](https://github.com/ZYK100/LLCM/blob/main/imgs/img2.png)

### Results
#### We have made some updates to the results in the our paper on the LLCM dataset. Please cite the results in the table below.

|Methods    | Rank@1   | Rank@10   | Rank@20   | mAP     | Rank@1   | Rank@10   | Rank@20   | mAP     |
| --------   | -----    |  -----  | -----    |  -----  | -----    |  -----  | -----    |  -----  |
|[DDAG](https://github.com/mangye16/DDAG)      | 42.36%  | 72.69% | 80.63%  | 48.97% | 51.42 %  | 81.45% | 88.26%  | 38.77% |
|[CMAlign](https://github.com/cvlab-yonsei/LbA)  | 42.76%  | 77.40% | 86.11%  | 50.95% | 54.78%  | 85.12% | 91.63%  | 40.81% |
|[AGW](https://github.com/mangye16/Cross-Modal-Re-ID-baseline)  | 49.13%  | 79.06% | 85.89%  | 55.80% | 63.72%  | 88.66% | 92.83%  | 47.21% |
|[CAJ](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/tree/master/ICCV21_CAJ)  | 49.86%  | 78.91% | 85.83%  | 56.40% | 63.73%  | 87.95% | 92.41%  | 47.71% |
|[MMN](https://github.com/ZYK100/MMN)  | 50.14%  | 79.81% | 87.27%  | 56.66% | 63.97%  | 88.66% | 93.05%  | 48.47% |
|MRCN  | 51.32%  | 80.10% | 87.17%  | 57.74% | 65.27%  | 88.11% | 93.13%  | 49.45% |
|[DART](https://github.com/XLearning-SCU/2022-CVPR-DART)  | 52.97%  | 80.82% | 87.05%  | 59.28% | 65.33%  | 89.42% | 93.33%  | 51.13% |
|DEEN (ours)  | 55.52%  | 83.88% | 89.98%  | 62.07% | 69.21%  | 90.95% | 95.07%  | 55.52% |

The results may have some fluctuation due to random spliting and it might be better by finetuning the hyper-parameters.

### Visualization

![tsne](https://github.com/ZYK100/LLCM/blob/main/Visualization/imgs/tsne_0.jpg)

### Citation
If you use the dataset, please cite the following paper:
```
  @InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Yukang and Wang, Hanzi},
    title     = {Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {2153-2162}
}
```

### Contact
If you have any question, please feel free to contact us. E-mail: zhangyk@stu.xmu.edu.cn

