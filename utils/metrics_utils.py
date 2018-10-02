import os
import sys
import tensorflow as tf
from os.path import join, exists, isdir, dirname, abspath, basename

from utils.tf_ops.cd import tf_nndistance
from utils.tf_ops.emd.tf_auctionmatch import auction_match

def scale(gt_pc, pr_pc): #pr->[-0.5,0.5], gt->[-0.5,0.5]
	'''
	Scale the input point clouds between [-max_length/2, max_length/2]
	'''
	# Cast the point cloud to float32
	pred = tf.cast(pr_pc, dtype=tf.float32) #(B, N, 3)
	gt   = tf.cast(gt_pc, dtype=tf.float32) #(B, N, 3)

	# Calculate min and max along each axis x,y,z for all point clouds in the batch
	min_gt = tf.convert_to_tensor([tf.reduce_min(gt[:,:,i], axis=1) for i in xrange(3)]) #(3, B)
	max_gt = tf.convert_to_tensor([tf.reduce_max(gt[:,:,i], axis=1) for i in xrange(3)]) #(3, B)
	min_pr = tf.convert_to_tensor([tf.reduce_min(pred[:,:,i], axis=1) for i in xrange(3)]) #(3, B)
	max_pr = tf.convert_to_tensor([tf.reduce_max(pred[:,:,i], axis=1) for i in xrange(3)]) #(3, B)

	# Calculate x,y,z dimensions of bounding cuboid
	length_gt = tf.abs(max_gt - min_gt) #(3, B)
	length_pr = tf.abs(max_pr - min_pr) #(3, B)

	# Calculate the side length of bounding cube (maximum dimension of bounding cuboid)
	# Then calculate the delta between each dimension of the bounding cuboid and the side length of bounding cube
	diff_gt = tf.reduce_max(length_gt, axis=0, keep_dims=True) - length_gt #(3, B)
	diff_pr = tf.reduce_max(length_pr, axis=0, keep_dims=True) - length_pr #(3, B)

	# Pad the xmin, xmax, ymin, ymax, zmin, zmax of the bounding cuboid to match the cuboid side length
	new_min_gt = tf.convert_to_tensor([min_gt[i,:] - diff_gt[i,:]/2. for i in xrange(3)]) #(3, B)
	new_max_gt = tf.convert_to_tensor([max_gt[i,:] + diff_gt[i,:]/2. for i in xrange(3)]) #(3, B)
	new_min_pr = tf.convert_to_tensor([min_pr[i,:] - diff_pr[i,:]/2. for i in xrange(3)]) #(3, B)
	new_max_pr = tf.convert_to_tensor([max_pr[i,:] + diff_pr[i,:]/2. for i in xrange(3)]) #(3, B)

	# Compute the side length of bounding cube
	size_pr = tf.reduce_max(length_pr, axis=0) #(B,)
	size_gt = tf.reduce_max(length_gt, axis=0) #(B,)

	# Calculate scaling factor according to scaled cube side length (here = 2.)
	scaling_factor_gt = 1. / size_gt #(B,)
	scaling_factor_pr = 1. / size_pr #(B,)

	# Calculate the min x,y,z coordinates for the scaled cube (here = (-1., -1., -1.))
	box_min = tf.ones_like(new_min_gt) * -0.5 #(3, B)

	# Calculate the translation adjustment factor to match the minimum coodinates of the scaled cubes
	adjustment_factor_gt = box_min - scaling_factor_gt * new_min_gt #(3, B)
	adjustment_factor_pr = box_min - scaling_factor_pr * new_min_pr #(3, B)

	# Perform scaling then translation to the point cloud ? verify this
	pred_scaled = tf.transpose((tf.transpose(pred) * scaling_factor_pr)) + tf.reshape(tf.transpose(adjustment_factor_pr), (-1,1,3))
	gt_scaled   = tf.transpose((tf.transpose(gt) * scaling_factor_gt)) + tf.reshape(tf.transpose(adjustment_factor_gt), (-1,1,3))

	return gt_scaled, pred_scaled

def get_chamfer_metrics(pcl_gt, pred):
	'''
	Calculate chamfer, forward, backward distance between ground truth and predicted
	point clouds. They may or may not be scaled.
	Args:
		pcl_gt: tf placeholder of shape (B,N,3) corresponding to GT point cloud
		pred: tensor of shape (B,N,3) corresponding to predicted point cloud
	Returns:
		Fwd, Bwd, Chamfer: (B,)
	'''
	#(B, NUM_POINTS) ==> ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2) for nn pair of points (x1,y1,z1) <--> (x2, y2, z2)
	dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pcl_gt, pred)
	dists_forward = tf.reduce_mean(tf.sqrt(dists_forward), axis=1) # (B, )
	dists_backward = tf.reduce_mean(tf.sqrt(dists_backward), axis=1) # (B, )
	chamfer_distance = dists_backward + dists_forward
	return dists_forward, dists_backward, chamfer_distance

def get_emd_metrics(pcl_gt, pred, batch_size, num_points):
	'''
	Calculate emd between ground truth and predicted point clouds. 
	They may or may not be scaled. GT and pred need to be of the same dimension.
	Args:
		pcl_gt: tf placeholder of shape (B,N,3) corresponding to GT point cloud
		pred: tensor of shape (B,N,3) corresponding to predicted point cloud
	Returns:
		emd: (B,)
	'''
	X,_ = tf.meshgrid(range(batch_size), range(num_points), indexing='ij')
	ind, _ = auction_match(pred, pcl_gt) # Ind corresponds to points in pcl_gt
	ind = tf.stack((X, ind), -1)
	emd = tf.reduce_mean(tf.sqrt(tf.reduce_sum((tf.gather_nd(pcl_gt, ind) - pred)**2, axis=-1)), axis=1) # (B, )
	return emd

def log_values(path, _chamfer, _fwd, _bwd, _emd):
	'''
	Log metrics in csv file
	'''
	with open(path, 'a') as f:
		f.write('{:.6f};{:.6f};{:.6f};{:.6f}\n'.format(_chamfer, _fwd, _bwd, _emd))
	return
