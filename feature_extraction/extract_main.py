from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from feature_extraction.extract_features import run
from feature_extraction.models.resnet import i3_res50
import os


def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode):
	feature_list = []
	Path(outputpath).mkdir(parents=True, exist_ok=True)
	temppath = outputpath+ "/temp/"
	rootdir = Path(datasetpath)
	videos = [str(f) for f in rootdir.glob('**/*.avi')]
	# setup the model
	i3d = i3_res50(400, pretrainedpath)
	i3d.cuda()
	i3d.train(False)  # Set model to evaluate mode
	for video in videos:
		videoname = video.split("/")[-1].split(".")[0]
		startime = time.time()
		print("Generating for {0}".format(video))
		Path(temppath).mkdir(parents=True, exist_ok=True)
		ffmpeg.input(video).output('{}%d.jpg'.format(temppath),start_number=0).global_args('-loglevel', 'quiet').run()
		print("Preprocessing done..")
		features = run(i3d, frequency, temppath, batch_size, sample_mode)
		feature_list.append(features)
		np.save(outputpath + "/" + videoname, features)
		print("Obtained features of size: ", features.shape)
		shutil.rmtree(temppath)
		print("done in {0}.".format(time.time() - startime))
	return feature_list

if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasetpath', type=str, default="/home/hienvx1/Documents/ActionSegment/rgb/augmented_rgb/")
	parser.add_argument('--outputpath', type=str, default="extract_aug_videos")
	parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
	parser.add_argument('--frequency', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--sample_mode', type=str, default="center_crop")
	args = parser.parse_args()
	generate(args.datasetpath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode)    
