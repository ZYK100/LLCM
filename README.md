### Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification

[Paper](https://arxiv.org/abs/2303.14481) (CVPR 2023).

### Abstract

For the visible-infrared person re-identification (VIReID) task, one of the major challenges is the modality gaps between visible (VIS) and infrared (IR) images. However, the training samples are usually limited, while the modality gaps are too large, which leads that the existing methods cannot effectively mine diverse cross-modality clues. To handle this limitation, we propose a novel augmentation network in the embedding space, called diverse embedding expansion network (DEEN). The proposed DEEN can effectively generate diverse embeddings to learn the informative feature representations and reduce the modality discrepancy between the VIS and IR images. Moreover, the VIReID model may be seriously affected by drastic illumination changes, while all the existing VIReID datasets are captured under sufficient illumination without significant light changes. Thus, we provide a low-light cross-modality (LLCM) dataset, which  contains 46,767 bounding boxes of 1,064 identities captured by 9 RGB/IR cameras. Extensive experiments on the SYSU-MM01, RegDB and LLCM datasets show the superiority of the proposed DEEN over several other state-of-the-art methods. 
![image](https://github.com/ZYK100/LLCM/blob/main/imgs/img1.png)

### Dataset download:
Please send a signed [dataset release agreement](https://github.com/ZYK100/LLCM/blob/main/Agreement/LLCM%20DATASET%20RELEASE%20AGREEMENT.pdf) copy to zhangyk@stu.xmu.edu.cn. If your application is passed, we will send the download link of the dataset.

![image](https://github.com/ZYK100/LLCM/blob/main/imgs/img2.png)

### The code will be released soon.

### Visualization

![tsne](https://github.com/ZYK100/LLCM/blob/main/Visualization/imgs/tsne_0.jpg)

### Citation
If you use the dataset, please cite the following paper:
```
@misc{zhang2023diverse,
      title={Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification}, 
      author={Yukang Zhang and Hanzi Wang},
      year={2023},
      eprint={2303.14481},
      archivePrefix={arXiv}
}
```

### Contact
If you have any question, please feel free to contact us. E-mail: zhangyk@stu.xmu.edu.cn

