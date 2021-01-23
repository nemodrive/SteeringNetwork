# recrops the dataset and adds the camera calibration file for each video frames directory
import shutil
import cv2
import json
import numpy as np
import os


def crop_to_size(img, img_height, img_width):
    og_height, og_width, _ = img.shape
    top = (og_height - img_height) // 2
    bot = og_height - (og_height - img_height) // 2
    left = (og_width - img_width) // 2
    right = og_width - (og_width - img_width) // 2
    img = img[top:bot, left:right]
    return img




raw_path = '/mnt/storage/workspace/andreim/nemodrive/sessions/'
days = os.listdir(raw_path)

for split in ['train', 'test', 'validation']:
	dataset_path = '/mnt/storage/workspace/andreim/nemodrive/upb_data/dataset/{}_frames/'.format(split)
	sessions = os.listdir(dataset_path)
	gb = ['good']

	cam = {}

	# find center_full movies and get the camera matrix to each one of them
	for day in days:
		day_path = os.path.join(raw_path, day)
		export_sessions = os.listdir(day_path)
		for es in export_sessions:
			for driving in gb:
				es_path = os.path.join(day_path, es, driving)
				files = os.listdir(es_path)
				for f in files:
					if 'center_full.mov' in f or '-0' in f:
						json_file = os.path.join(es_path, f.replace('mov', 'json').replace('-0', ''))
						with open(json_file, 'r') as jf:
							json_data = json.load(jf)
							camera_matrix = np.array(json_data['cameras'][0]['cfg_extra']['camera_matrix'])
							camera_matrix[0, :] /= 3.0
							camera_matrix[1, :] /= 3.0
							camera_matrix[1, 2] = camera_matrix[1, 2] / 360 * 288
							cam[f.replace('-0.mov', '')] = camera_matrix

	print(cam.keys())
	# go through sessions
	for s in sessions:
		print(s)
		session_path = os.path.join(dataset_path, s)
		frames = os.listdir(session_path)
		for frame in frames:
			print(frame, frame[5:-4])
			frame_no = int(frame[5:-4])
			frame_path = os.path.join(session_path, frame)
			img = cv2.imread(frame_path)
			height, width, _ = img.shape
			img = crop_to_size(img, height - int(0.2 * height), width)
			os.remove(frame_path)
			cv2.imwrite(os.path.join(session_path, '{0:06d}.png'.format(frame_no)), img)
		np.savetxt(os.path.join(session_path, 'cam.txt'), cam[s])
