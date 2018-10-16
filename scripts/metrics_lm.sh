#!/bin/bash

gpu=0
exp=trained_models/lm
eval_set=valid
data_dir_imgs=data/ShapeNetRendering
data_dir_pcl=data/ShapeNet_pointclouds
declare -a categs=("airplane" "bench" "cabinet" "car" "chair" "lamp" "monitor" "rifle" "sofa" "speaker" "table" "telephone" "vessel")

for cat in "${categs[@]}"; do
	echo python metrics.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --mode lm --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24
	python metrics.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --mode lm --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24
done

declare -a categs=("airplane" "bench" "cabinet" "car" "chair" "lamp" "monitor" "rifle" "sofa" "speaker" "table" "telephone" "vessel")
for cat in "${categs[@]}"; do
	echo ${cat}
	cat ${exp}/metrics/${eval_set}/${cat}.csv
	echo
done