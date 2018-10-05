import os
from os import listdir
from os.path import exists, join
import shutil
import sys
import json
from utils.shapenet_taxonomy import shapenet_id_to_category

src_path = '/data/val/navaneet/3DR/3DRModels/ShapeNet'
dst_path = '/data/val/navaneet/3DR/3DRModels/ShapeNet_validation'

if __name__=='__main__':

	with open(join('data/', 'val_models.json'), 'r') as f:
		val_models_dict = json.load(f)

	val_models = []
	cats = shapenet_id_to_category.keys()

	for cat in cats:
		val_models.extend([model for model in val_models_dict[cat]])

	total = len(val_models)
	print 'Preparing to copy %d models'%total
	count = 0

	for model in val_models:

		model_src_path = join(src_path, model)
		model_dst_path = join(dst_path, model)

		if not exists(join(model_dst_path, 'rendering')):
			  shutil.copytree(join(model_src_path, 'rendering'), join(model_dst_path, 'rendering'))

		src_1k = join(model_src_path, 'pointcloud_trimesh_fps_1024.npy')
		dst_1k = join(model_dst_path, 'pointcloud_trimesh_fps_1024.npy')
		
		if exists(src_1k):
				if not exists(dst_1k):
						shutil.copy(src_1k, dst_1k)
		count += 1
		if count%100 == 0:
			print ('Copied %d / %d'% (count, total))

		# break


