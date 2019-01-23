#!/bin/bash

gpu=0
exp=trained_models/lm
eval_set=valid
dataset=shapenet
data_dir_imgs=data/shapenet/ShapeNetRendering
data_dir_pcl=data/shapenet/ShapeNet_pointclouds
declare -a categs=("airplane" "bench" "cabinet" "car" "chair" "lamp" "monitor" "rifle" "sofa" "speaker" "table" "telephone" "vessel")

for cat in "${categs[@]}"; do
	python metrics.py \
		--gpu $gpu \
		--dataset $dataset \
		--data_dir_imgs ${data_dir_imgs} \
		--data_dir_pcl ${data_dir_pcl} \
		--exp $exp \
		--mode lm \
		--category $cat \
		--load_best \
		--bottleneck 512 \
		--bn_decoder \
		--eval_set ${eval_set} \
		--batch_size 24
done

declare -a categs=("airplane" "bench" "cabinet" "car" "chair" "lamp" "monitor" "rifle" "sofa" "speaker" "table" "telephone" "vessel")
for cat in "${categs[@]}"; do
	echo ${cat}
	cat ${exp}/metrics_$dataset/${eval_set}/${cat}.csv
	echo
done