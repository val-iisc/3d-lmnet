python train_lm.py \
	--exp expts/lm_all_cat \
	--gpu 0 \
	--ae_logs expts/ae_all_cat \
	--category all \
	--bottleneck 512 \
	--loss l1 \
	--batch_size 32 \
	--lr 5e-5 \
	--bn_decoder \
	--load_best_ae \
	--max_epoch 50 \
	--print_n 100
	# --sanity_check
