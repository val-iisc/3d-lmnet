#!/bin/bash

gpu=0
exp=trained_models/plm
data_dir_imgs=data/ShapeNetRendering
data_dir_pcl=data/ShapeNet_pointclouds
eval_set=valid
cat=chair

echo python metrics.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --mode plm --category ${cat} --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24
python metrics.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --mode plm --category ${cat} --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24
cat ${exp}/metrics/${eval_set}/${cat}.csv