#!/bin/bash

gpu=0
exp=trained_models/lm
eval_set=valid
declare -a categs=("airplane" "bench" "cabinet" "car" "chair" "lamp" "monitor" "rifle" "sofa" "speaker" "table" "telephone" "vessel")

for cat in "${categs[@]}"; do
	echo python metrics.py --gpu $gpu --exp $exp --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24
	python metrics.py --gpu $gpu --exp $exp --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24
done

for cat in "${categs[@]}"; do
	echo ${cat}
	cat ${exp}/metrics/${eval_set}/${cat}.csv
	echo
done