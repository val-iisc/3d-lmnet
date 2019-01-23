import json
import os
import re
import sys
import numpy as np
from itertools import product
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename

from utils.shapenet_taxonomy import shapenet_id_to_category, shapenet_category_to_id

NUM_VIEWS = 24
PNG_FILES = [(str(i).zfill(2)+'.png') for i in xrange(NUM_VIEWS)]

def create_folder(folder):
	if not exists(folder):
		makedirs(folder)

def create_folders(folders):
	for folder in folders:
		create_folder(folder)

def get_shapenet_models(FLAGS):
	'''
	Training and validation set creation
	Args:
		FLAGS: arguments parsed for the particular experiment
	Returns:
		train_models: list of models (absolute path to each model) in the training set
		val_models:	list of models (absolute path to each model) in the validation set
		train_pair_indices: list of ind pairs for training set
		val_pair_indices: list of ind pairs for validation set
		-->	ind[0] : model index (range--> [0, len(models)-1])
		-->	ind[1] : view index (range--> [0, NUM_VIEWS-1])
	'''

	with open(join('data/splits/', 'train_models.json'), 'r') as f:
		train_models_dict = json.load(f)

	with open(join('data/splits/', 'val_models.json'), 'r') as f:
		val_models_dict = json.load(f)

	train_models = []
	val_models = []

	category = FLAGS.category
	if category == 'all':
		cats = shapenet_id_to_category.keys()
	else:
		cats = [shapenet_category_to_id[category]]
	
	for cat in cats:
		train_models.extend([model for model in train_models_dict[cat]])

	for cat in cats:
		val_models.extend([model for model in val_models_dict[cat]])

	train_pair_indices = list(product(xrange(len(train_models)), xrange(NUM_VIEWS)))
	val_pair_indices = list(product(xrange(len(val_models)), xrange(NUM_VIEWS)))

	print 'TRAINING: models={}  samples={}'.format(len(train_models),len(train_models)*NUM_VIEWS)
	print 'VALIDATION: models={}  samples={}'.format(len(val_models),len(val_models)*NUM_VIEWS)
	print

	return train_models, val_models, train_pair_indices, val_pair_indices


def get_pix3d_models(FLAGS):

	with open(join(FLAGS.data_dir_imgs, 'pix3d.json'), 'r') as f:
		models_dict = json.load(f)
	models = []

	if FLAGS.category == 'all':
		cats = ['chair','sofa','table']
	else:
		cats = [FLAGS.category]
	
	# Check for truncation and occlusion before adding a model to the evaluation list
	for d in models_dict:
		if d['category'] in cats:
			if not d['truncated'] and not d['occluded'] and not d['slightly_occluded']:
				models.append(d)

	print 'Total models = {}\n'.format(len(models))
	return models


def rotate(xyz, xangle=0, yangle=0, zangle=0):
	rotmat = np.eye(3)

	rotmat=rotmat.dot(np.array([
		[1.0,0.0,0.0],
		[0.0,np.cos(xangle),-np.sin(xangle)],
		[0.0,np.sin(xangle),np.cos(xangle)],
		]))

	rotmat=rotmat.dot(np.array([
		[np.cos(yangle),0.0,-np.sin(yangle)],
		[0.0,1.0,0.0],
		[np.sin(yangle),0.0,np.cos(yangle)],
		]))

	rotmat=rotmat.dot(np.array([
		[np.cos(zangle),-np.sin(zangle),0.0],
		[np.sin(zangle),np.cos(zangle),0.0],
		[0.0,0.0,1.0]
		]))

	return xyz.dot(rotmat)