# Binary Classification

This is the code for training and evaluation using binary classifiers.
We only provide scripts for training on CLEVR datasets. Please let me know if other datasets
are also needed.

## Training
Training binary classifiers on CLEVR Relational dataset.
```
python train.py --spec_norm --norm --im_size 128 --batch_size 32 --dataset clevr_rel --lr 1e-5 --checkpoint_dir results
```
Training binary classifiers on CLEVR 2D Position dataset.
```
python train.py --spec_norm --norm --im_size 128 --batch_size 32 --dataset clevr_pos --lr 1e-5 --checkpoint_dir results
```

## Evaluation
Evaluation using generated samples on CLEVR Relational dataset.
```
python eval.py --dataset clevr_rel --checkpoint_dir results --im_size 128 --filter_dim 64  --npy_path $GROUND_TRUTH_NPY_PATH  --generated_img_folder $IMG_PATH --mode generation
```
Evaluation using generated samples on CLEVR 2D Position dataset.
```
python eval.py --dataset clevr_pos --checkpoint_dir results --im_size 128 --filter_dim 64  --npy_path $GROUND_TRUTH_NPY_PATH  --generated_img_folder $IMG_PATH --mode generation
```
<hr>


## Datasets
|                    Dataset                    | Link | 
|:---------------------------------------------:| :---: | 
|  CLEVR Relations Classifier Training Dataset  | https://www.dropbox.com/s/ptqtd0j771sgs7h/clevr_rel_128_30000.pickle?dl=0|
| CLEVR 2D Position Classifier Training Dataset | https://www.dropbox.com/s/7etqyrwg9g854t4/clevr_pos_data_128_30000.npz?dl=0|
|         CLEVR Relation Test Dataset 1         | https://www.dropbox.com/s/bfj4wjb4ksic6z2/clevr_generation_1_relations.npz?dl=0|
|         CLEVR Relation Test Dataset 2         | https://www.dropbox.com/s/g59mscx6880j72h/clevr_generation_2_relations.npz?dl=0|
|         CLEVR Relation Test Dataset 3         | https://www.dropbox.com/s/nvc2mdsixi7vu3i/clevr_generation_3_relations.npz?dl=0|
|         CLEVR 2D Position Test Dataset 1         | https://www.dropbox.com/s/je1nbw463ic1urm/clevr_pos_5000_1.npz?dl=0|
|         CLEVR 2D Position Test Dataset 2         | https://www.dropbox.com/s/8svd75vw1j7wmzj/clevr_pos_5000_2.npz?dl=0|
|         CLEVR 2D Position Test Dataset 3         | https://www.dropbox.com/s/j3d32udsg1cewvc/clevr_pos_5000_3.npz?dl=0|

## Pretrained Classifiers
|      Models       | Link | 
|:-----------------:| :---: | 
|  CLEVR Relations  | https://www.dropbox.com/s/a0ya3yo4nfgg8c8/232.tar?dl=0|
| CLEVR 2D Position | https://www.dropbox.com/s/kf86h2dp5zasqtq/544.tar?dl=0|

Before running evaluation script, download pretrained classifiers to following folder:
```
./classifiers/clevr_rel_classifier_128
./classifiers/clevr_pos_classifier_128
```