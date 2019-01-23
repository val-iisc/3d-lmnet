python train_plm.py \
	--mode lm \
	--data_dir_imgs data/shapenet/ShapeNetRendering \
	--data_dir_pcl data/shapenet/ShapeNet_pointclouds \
	--exp expts/plm_chairs \
	--gpu 1 \
	--ae_logs expts/ae_all_cat \
	--category chair \
	--bottleneck 512 \
	--loss vae \
	--batch_size 32 \
	--lr 5e-5 \
	--bn_decoder \
	--load_best_ae \
	--max_epoch 50 \
	--print_n 100 \
	--alpha 0.2 \
	--penalty_angle 20 \
	--weight 5.5
	# --sanity_check
