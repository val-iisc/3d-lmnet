import os
import re
import sys
import tensorflow as tf
from os.path import join

def load_previous_checkpoint(snapshot_folder, saver, sess, is_training=True):
	'''
	Load previous latent matching snapshot
	'''
	start_epoch = 0
	ckpt = tf.train.get_checkpoint_state(snapshot_folder)
	if ckpt is not None:
		ckpt_path = ckpt.model_checkpoint_path
		ckpt_path = join(snapshot_folder, ckpt_path.split('/')[-1])
		print ('loading '+ckpt_path + '  ....')
		saver.restore(sess, ckpt_path)
		if is_training:
			start_epoch = 1 + int(re.match('.*-(\d*)$', ckpt_path).group(1))
			return start_epoch
	else:
		if is_training:
			return start_epoch
		else:
			print 'Snapshot not found! Please check the path:'
			print snapshot_folder
			sys.exit(1)

def load_pointnet_ae(pointnet_ae_logs_path, pointnet_ae_vars, sess, FLAGS):
	'''
	Load Pretrained autoencoder model 
	'''
	saver_pointnet_ae = tf.train.Saver(pointnet_ae_vars)

	try:
		if FLAGS.load_best_ae:
			ckpt_path = join(pointnet_ae_logs_path, 'best', 'best')
			print ('loading ' + ckpt_path)
			saver_pointnet_ae.restore(sess, ckpt_path)
		else:
			snapshot_folder = join(pointnet_ae_logs_path,'snapshots')
			ckpt = tf.train.get_checkpoint_state(snapshot_folder)
			ckpt_path = ckpt.model_checkpoint_path
			ckpt_path = join('..', '/'.join(ckpt_path.split('/')[-4:]))
			print ('loading '+ckpt_path + '  ....')
			saver_pointnet_ae.restore(sess, ckpt_path)
	except Exception as e:
		print e
		sys.exit('No PointNet checkpoint loaded!')

	return