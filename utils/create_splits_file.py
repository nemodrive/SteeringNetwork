# starting from the training and validation directories, creates train and validation splits files for them
import os


dataset_path = '/raid/workspace/andreim/nemovrive/upb_data/dataset/'
train_path = dataset_path + 'train_frames/'
val_path = dataset_path + 'validation_frames/'
sessions_train = os.listdir(train_path)
sessions_val = os.listdir(val_path)

train_file = dataset_path + 'train.txt'
val_file = dataset_path + 'val.txt'

with open(train_file, 'w+') as f:
	for s in sessions_train:
		if '.txt' not in s:
			f.write(train_path + s + '\n')

with open(val_file, 'w+') as f:
        for s in sessions_val:
                if '.txt' not in s:
                        f.write(val_path + s + '\n')
