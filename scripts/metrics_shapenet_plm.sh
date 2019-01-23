#!/bin/bash

gpu=0
exp=trained_models/plm
dataset=shapenet
data_dir_imgs=data/shapenet/ShapeNetRendering
data_dir_pcl=data/shapenet/ShapeNet_pointclouds
eval_set=valid
cat=chair

python metrics.py \
	--gpu $gpu \
	--dataset $dataset \
	--data_dir_imgs ${data_dir_imgs} \
	--data_dir_pcl ${data_dir_pcl} \
	--exp $exp \
	--mode plm \
	--category ${cat} \
	--load_best \
	--bottleneck 512 \
	--bn_decoder \
	--eval_set ${eval_set} \
	--batch_size 24
	
cat ${exp}/metrics_$dataset/${eval_set}/${cat}.csv