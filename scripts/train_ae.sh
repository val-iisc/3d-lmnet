python train_ae.py \
	--exp expts/ae_all_cat \
	--gpu 1 \
	--category all \
	--bottleneck 512 \
	--batch_size 32 \
	--lr 5e-4 \
	--bn_decoder \
	--print_n 100 \
