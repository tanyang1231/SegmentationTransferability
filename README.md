## [Efficient Prediction of Model Transferability in Semantic Segmentation Tasks](http://yangli-feasibility.com/home/media/icip-23.pdf)



### 1. Preprare Environment 
Install python libraries following the ``requirements.txt``

```
pip install -r requirements.txt
```
Then download extracted features from google cloud via link and unzip it in this project directory. As the features for all transfer task pairs are too large to share. Here we only select one task pair ``segnet-trained-on-BDD100K -> aachen (from Cityscapes)``as example.

The ``feature`` directory contains ``source_feature`` and ``target_feature``, which are extracted from the source model ``segnet-trained-on-BDD100K`` by feeding the source and target data into the neural network, respectively. Here we keep the feature of the final layer.

### 2. Computing Model Transferability

To compute the transferability score of the source model on a given target task. Jump into ``computing_model_transferability`` directory and run:
```
bash compute_xxx.sh
```
to produce transferability scores. We support transferability metrics OTCE, LEEP, LogME and H-score.

### 3. Computing Pixel-wise Transferability

In addition to computing model transferability, we also provide the code to compute pixel-wise (patch-wise) transferability for analyzing detailed transferability in the local regions on images. 

Jump into ``compute_transferability_map`` directory and run:

```
bash compute_pixelwise_transferability.sh
```
You can modify the hyperparameter ``metric`` for generating transferability maps for different metrics. ``stride`` controls the patch size. The generated transferability maps (``.png`` and ``.npy``) will be saved in ``trf_maps``.

### 4. Acknowledgement

If you find our work interesting, please consider citing our work^_^.
```
@inproceedings{tan2023efficient,
  title={Efficient Prediction of Model Transferability in Semantic Segmentation Tasks},
  author={Tan, Yang and Li, Yicong and Li, Yang and Zhang, Xiao-Ping},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={720--724},
  year={2023},
  organization={IEEE}
}
```
### 5. Interesting Transfer Learning Papers from Our Group

```
1. OTCE: A Transferability Metric for Cross-Domain Cross-Task Representations (CVPR2021)

2. Transferability-Guided Cross-Domain Cross-Task Transfer Learning (IEEE-TNNLS 2024)

3. Finding the Most Transferable Tasks for Brain Image Segmentation (BIBM2022)

4. Investigating Consistency Constraints in Heterogeneous Multi-task Learning for Medical Image Processing (BIBM2023)

5. An Information-Theoretic Approach to Transferability in Task Transfer Learning (ICIP2019)

6. H-ensemble: An Information Theoretic Approach to Reliable Few-Shot Multi-Source-Free Transfer (AAAI2023)

7. Enhancing Continuous Domain Adaptation with Multi-Path Transfer Curriculum (PAKDD2024)

8. Joint PVL Detection and Manual Ability Classification Using Semi-supervised Multi-task Learning (MICCAI2021)

9. Distributionally Robust Domain Generalization (ICLR Workshop 2021)
```
