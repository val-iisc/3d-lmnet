import os
import sys
import tensorflow as tf
from os.path import join, exists, isdir, dirname, abspath, basename

from utils.tf_ops import tf_nndistance

def scale(pcl, max_length=1.):
	'''
	Scale the input point cloud between [-max_length/2, max_length/2]
	Args:
		pcl: point cloud to be scaled (B,N,3)
		max_length: length of bounding box that holds the point cloud
	Returns:
		pcl: scaled point cloud according to max_length (B,N,3)
	'''
	bound_l = tf.reduce_min(pcl, axis=1, keep_dims=True)
	bound_h = tf.reduce_max(pcl, axis=1, keep_dims=True)
	pcl = pcl - (bound_l + bound_h) / 2.
	pcl = pcl / tf.reduce_max((bound_h - bound_l), axis=[1,2], keep_dims=True)
	pcl = pcl*max_length
	return pcl

def get_chamfer_metrics(pcl_gt, reconstr):
	'''
	Calculate chamfer, forward, backward distance between ground truth and predicted
	point clouds. They may or may not be scaled.
	Args:
		pcl_gt: tf placeholder of shape (B,N,3) corresponding to GT point cloud
		reconstr: tensor of shape (B,N,3) corresponding to predicted point cloud
	'''
	dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pcl_gt, reconstr)
	dists_forward = tf.reduce_mean(dists_forward)
	dists_backward = tf.reduce_mean(dists_backward)
	chamfer_distance = dists_backward + dists_forward
	return dists_forward, dists_backward, chamfer_distance

