from importer import *

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir_pcl', type=str, required=True, 
	help='Path to shapenet pointclouds')
parser.add_argument('--exp', type=str, required=True, 
	help='Name of Experiment')
parser.add_argument('--gpu', type=str, required=True, 
	help='GPU to use')
parser.add_argument('--category', type=str, required=True, 
	help='Category to train on : \
		["all", "airplane", "bench", "cabinet", "car", "chair", "lamp", \
		"monitor", "rifle", "sofa", "speaker", "table", "telephone", "vessel"]')
parser.add_argument('--bottleneck', type=int, required=True, default=512, 
	help='latent space size')
parser.add_argument('--batch_size', type=int, default=32, 
	help='Batch Size during training')
parser.add_argument('--lr', type=float, default=0.0005, 
	help='Learning Rate')
parser.add_argument('--max_epoch', type=int, default=500, 
	help='max num of epoch')
parser.add_argument('--bn_decoder', action='store_true', 
	help='Supply this parameter if you want batch norm in the decoder, otherwise ignore')
parser.add_argument('--print_n', type=int, default=100, 
	help='print output to terminal every n iterations')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size  		# Batch size for training
NUM_POINTS = 2048					# Number of predicted points
GT_PCL_SIZE = 16384					# Number of points in GT point cloud


def fetch_batch(models, batch_num, batch_size):
	'''
	Input:
		models: list of paths to shapenet models
		batch_num: batch_num during epoch
		batch_size:	batch size for training or validation
	Returns:
		batch_gt: (B,2048,3)
	Description:
		Batch Loader
	'''

	batch_gt = []
	for ind in range(batch_num*batch_size, batch_num*batch_size+batch_size):
		model_path = models[ind]
		pcl_path = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_2048.npy') # Path to 2K ground truth point cloud
		pcl_gt = np.load(pcl_path)
		batch_gt.append(pcl_gt)
	batch_gt = np.array(batch_gt)
	return batch_gt

def get_epoch_loss(val_models):

	'''
	Input:
		val_models:	list of absolute path to models in validation set
	Returns:
		val_chamfer: chamfer distance calculated on scaled prediction and gt
		val_forward: forward distance calculated on scaled prediction and gt
		val_backward: backward distance calculated on scaled prediction and gt
	Description:
		Calculate val epoch metrics
	'''
	
	tflearn.is_training(False, session=sess)

	batches = len(val_models)/BATCH_SIZE
	val_stats = {}
	val_stats = reset_stats(ph_summary, val_stats)

	for b in xrange(batches):
		batch_gt = fetch_batch(val_models, b, BATCH_SIZE)
		runlist = [loss, chamfer_distance_scaled, dists_forward_scaled, dists_backward_scaled]
		L,C,F,B = sess.run(runlist, feed_dict={pcl_gt:batch_gt})
		_summary_losses = [L, C, F, B]

		val_stats = update_stats(ph_summary, _summary_losses, val_stats, batches)

	summ = sess.run(merged_summ, feed_dict=val_stats)
	return val_stats[ph_dists_chamfer], val_stats[ph_dists_forward], val_stats[ph_dists_backward], summ


if __name__ == '__main__':

	# Create a folder for experiments and copy the training file
	create_folder(FLAGS.exp)
	train_filename = basename(__file__)
	os.system('cp %s %s'%(train_filename, FLAGS.exp))
	with open(join(FLAGS.exp, 'settings.txt'), 'w') as f:
		f.write(str(FLAGS)+'\n')

	# Create Placeholders
	pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, 3))

	# Generate Prediction
	bneck_size = FLAGS.bottleneck
	with tf.variable_scope('pointnet_ae') as scope:
		z = encoder_with_convs_and_symmetry(in_signal=pcl_gt, n_filters=[64,128,128,256,bneck_size], 
			filter_sizes=[1],
			strides=[1],
			b_norm=True,
			verbose=True,
			scope=scope
			)
		out = decoder_with_fc_only(z, layer_sizes=[256,256,np.prod([NUM_POINTS, 3])],
			b_norm=FLAGS.bn_decoder,
			b_norm_finish=False,
			verbose=True,
			scope=scope
			)
		out = tf.reshape(out, (BATCH_SIZE, NUM_POINTS, 3))

	# Scale output and gt for val losses
	pcl_gt_scaled, out_scaled = scale(pcl_gt, out)
	
	# Calculate Chamfer Metrics
	dists_forward, dists_backward, chamfer_distance = [tf.reduce_mean(metric) for metric in get_chamfer_metrics(pcl_gt, out)]

	# Calculate Chamfer Metrics on scaled prediction and GT
	dists_forward_scaled, dists_backward_scaled, chamfer_distance_scaled = [tf.reduce_mean(metric) for metric in get_chamfer_metrics(pcl_gt_scaled, out_scaled)]

	# Define Loss to optimize on
	loss = (dists_forward + dists_backward/2.0)*10000

	# Get Training Models
	train_models, val_models, _, _ = get_shapenet_models(FLAGS)
	batches = len(train_models) / BATCH_SIZE

	# Training Setings
	lr = FLAGS.lr
	optim = tf.train.AdamOptimizer(FLAGS.lr, beta1=0.9).minimize(loss)

	start_epoch = 0
	max_epoch = FLAGS.max_epoch

	# Define Log Directories
	snapshot_folder = join(FLAGS.exp, 'snapshots')
	best_folder = join(FLAGS.exp, 'best')
	logs_folder = join(FLAGS.exp, 'logs')	

	# Define Savers
	saver = tf.train.Saver(max_to_keep=2)

	# Define Summary Placeholders
	ph_loss = tf.placeholder(tf.float32, name='loss')
	ph_dists_chamfer = tf.placeholder(tf.float32, name='dists_chamfer')
	ph_dists_forward = tf.placeholder(tf.float32, name='dists_forward')
	ph_dists_backward = tf.placeholder(tf.float32, name='dists_backward')

	ph_summary = [ph_loss, ph_dists_chamfer, ph_dists_forward, ph_dists_backward]
	merged_summ = get_summary(ph_summary)

	# Create log directories
	create_folders([snapshot_folder, logs_folder, join(snapshot_folder, 'best'), best_folder])

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:

		train_writer = tf.summary.FileWriter(logs_folder+'/train', sess.graph_def)
		val_writer = tf.summary.FileWriter(logs_folder+'/val', sess.graph_def)

		sess.run(tf.global_variables_initializer())

		# Load Previous checkpoint
		start_epoch = load_previous_checkpoint(snapshot_folder, saver, sess)

		ind = 0
		best_val_loss = 10000000
		since = time.time()

		print '*'*30,'\n','Training Started !!!\n', '*'*30

		PRINT_N = FLAGS.print_n

		for i in xrange(start_epoch, max_epoch): 
			random.shuffle(train_models)
			stats = {}
			stats = reset_stats(ph_summary, stats)
			iter_start = time.time()

			tflearn.is_training(True, session=sess)

			for b in xrange(batches):
				global_step = i*batches + b + 1
				batch_gt = fetch_batch(train_models, b, BATCH_SIZE)

				runlist = [loss, chamfer_distance, dists_forward, dists_backward, optim]
				L, C, F, B, _ = sess.run(runlist, feed_dict={pcl_gt:batch_gt})
				_summary_losses = [L, C, F, B]

				stats = update_stats(ph_summary, _summary_losses, stats, PRINT_N)

				if global_step % PRINT_N == 0:
					summ = sess.run(merged_summ, feed_dict=stats)
					train_writer.add_summary(summ, global_step)
					till_now = time.time() - iter_start
					print 'Loss = {} Iter = {}  Minibatch = {} Time:{:.0f}m {:.0f}s'.format(
						stats[ph_loss], global_step, b, till_now//60, till_now%60
					)
					stats = reset_stats(ph_summary, stats)
					iter_start = time.time()

			print 'Saving Model ....................'
			saver.save(sess, join(snapshot_folder, 'model'), global_step=i)
			print '..................... Model Saved'

			val_epoch_chamfer, val_epoch_forward, val_epoch_backward, val_summ = get_epoch_loss(val_models)
			val_writer.add_summary(val_summ, global_step)

			time_elapsed = time.time() - since

			print '-'*65 + ' EPOCH ' + str(i) + ' ' + '-'*65
			print 'Val Chamfer: {:.8f}  Forward: {:.8f}  Backward: {:.8f}  Time:{:.0f}m {:.0f}s'.format(
				val_epoch_chamfer, val_epoch_forward, val_epoch_backward, time_elapsed//60, time_elapsed%60
			)
			print '-'*140
			print

			if (val_epoch_chamfer < best_val_loss):
				print 'Saving Best at Epoch %d ...............'%(i)
				saver.save(sess, join(snapshot_folder, 'best', 'best'))
				os.system('cp %s %s'%(join(snapshot_folder, 'best/*'), best_folder))
				best_val_loss = val_epoch_chamfer
				print '.............................Saved Best'
