# deletes frames with corresponding speed < threshold
import shutil
import cv2
import json
import numpy as np
import os
import pandas as pd
import csv


col = 'Unnamed: 0'
threshold = 1e-3


for split in ['train', 'test', 'validation']:
	info_path = '/mnt/storage/workspace/andreim/nemodrive/upb_data/dataset/{}/info/'.format(split)
	dataset_path = '/mnt/storage/workspace/andreim/nemodrive/upb_data/dataset/{}_frames/'.format(split)
	sessions = os.listdir(dataset_path)


	# go through sessions
	for s in sorted(sessions):
		print(s)
		session_path = os.path.join(dataset_path, s)
		frames = os.listdir(session_path)
		info_file = os.path.join(info_path, s + '-0.csv')
		df = pd.read_csv(info_file)
		#print(df)

		sorted_frames = sorted(frames)

		# for it, frame in enumerate(sorted_frames):
		# 	if '.txt' not in frame:
		# 		#print(it, df[df[col] == it]['linear_speed'].values[0])
		# 		if abs(df[df[col] == it]['linear_speed'].values[0]) < threshold:
		# 			# df = df[df[col] != it]
		# 			# print(os.path.join(session_path, frame), info_file)
		# 			frame_path = os.path.join(session_path, frame)
		# 			img = cv2.imread(frame_path)
		# 			# os.remove(frame_path)
		# 			print('deleting {}'.format(frame_path))

		# frames = os.listdir(session_path)
		# sorted_frames = sorted(frames)

		# for ind in range(len(df)):
		# 	# df[col].iloc[ind] = ind
		# 	frame_path = os.path.join(session_path, sorted_frames[ind])
			# print(frame_path, '/'.join(frame_path.split('/')[:-1]) + 'frame{0:06d}.png'.format(ind))
			# os.rename(frame_path, '/'.join(frame_path.split('/')[:-1]) + '/frame{0:06d}.png'.format(ind))
		# df.to_csv(info_file.replace('.csv', '-static-deleted.csv'))

		with open(info_file, 'r') as inp, open(info_file.replace('.csv', '-static-deleted.csv'), 'w') as out:
			writer = csv.writer(out)
			current_ind = 0
			for row in csv.reader(inp):
				ls = row[4]
				if ls == 'linear_speed' or abs(float(ls)) >= threshold:
					if row[0] != '':
						row[0] = current_ind
						current_ind += 1
					writer.writerow(row)
					print(current_ind, ls)