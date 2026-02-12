
from data_loaders_AgePred_Final import get_loader
from models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet

from logger import Logger
from sklearn import metrics as skmet
from torch.autograd import Variable
from torchvision.utils import save_image

import datetime
import json
import numpy as np
import pandas as pd
import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import glob

import sys
import argparse
#from robust_loss_pytorch import AdaptiveLossFunction


parser = argparse.ArgumentParser(description="Age Predication using 3D MRI Images (supporting NII, NII.GZ, NP & MGZ input files)")
parser.add_argument("--inList", type=str, help="Path to CSV file listing MRI Images and Ages")
parser.add_argument(
    "--model",
    type=str,
    help=(
        "Deep Learning model to be used for Age Prediction (i.e., \"BrainAge\", \"SkullAge\", \"HeadAge\"). "
        "Choose the model based on your input images; for example in case of head images as input, "
        "only use the HeadAge model. Choosing other models will lead to inconsistent predictions."
    ),
)
parser.add_argument("--ext",  type=str, default="nii", help="MRI Images Extension (i.e. NII, NII.GZ, NP & MGZ")
parser.add_argument("--out", type=str, default="Results_AgePred.csv", help="Output Name for Age Predication Results")

args = parser.parse_args()

print("\nInput List:", args.inList)
print("Age Predication Model:", args.model)
print("MRI Image Extension:", args.ext)
print(f"Results Filen Name: {args.out}.csv")
print("#"*20)

#### Input/Output Parsing

## inputs
if args.inList != None:
	input_csv = args.inList
else:
	print("Input list is not provided, Exiting!")
	sys.exit()

if args.model == None:
	print("Model Name not provided, Exiting!")
	print("Valid Models: BrainAge, SkullAge, HeadAge")
	sys.exit()

args_model = args.model.lower()

if args_model ==  "brainage":
	model_ckpt = "./Model_Weights/BrainAge.ckpt"
	
elif args_model ==  "skullage":
	model_ckpt = "./Model_Weights/SkullAge.ckpt"
	
elif args_model ==  "headage":
	model_ckpt = "./Model_Weights/HeadAge.ckpt"
	
else:
	print("Model name is not valid! Exiting!")
	print("Valid Models: BrainAge, SkullAge, HeadAge")
	sys.exit()

fileExt = args.ext

## Output
results_file = args.out + ".csv"

#### Input/Output Parsing

FEATS = []

# placeholder for batch features
features = {}

def get_features(name):
	def hook(model, input, output):
		features[name] = output.detach()
	return hook

metrics_f = {
	"r2_score": skmet.r2_score,
	"mae": skmet.mean_absolute_error,
}

class RMSLELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()
		
	def forward(self, pred, actual):
		return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def get_original_age(age, lower, upper):
	return lower + (upper - lower)*age

def enable_dropout(m):
	for each_module in m.modules():
		if each_module.__class__.__name__.startswith('Dropout'):
			each_module.train()

def val_test(model, dataset, criterion, device, cls_num=2, mode="val"):
	
	model.eval()
	enable_dropout(model)
	
	
	gt, gt_orig = [], []
	pred, pred_orig = [], []
	metrics = {}
	print("#"*20)
	with torch.no_grad():
		results_file_lastlayer = results_file.replace('.csv','_lastlayer.csv')
		resultsfilelastlayer = open(results_file_lastlayer, 'w')
		
		with open (results_file, "w") as resultfile:
			fieldnames = ['ID', 'Choronoligal Age', 'Predicted Age']
			reswriter = csv.DictWriter(resultfile, fieldnames=fieldnames)
			reswriter.writeheader()
			
			resultsfilelastlayer.write(','.join(fieldnames) + ', ,lastLayerValues' + '\n')
			
			#print(len(dataset))
			
			
			for x, y, z in dataset:		
				x = x.to(device)
				y = y.to(device)
				
				y_ = model(x)
				
				lastlayer = features["feats"].cpu().numpy()
				lastlayer_reshaped = lastlayer.reshape(-1)
				
				#print(f"y.item(): {y.item()}, y_.item(): {y_.item()}")
				#print(f"Choronoligal Age: {y.item()}, Predicted Age: {y_.item():.3f}")

				residual = (y_ - y.float())

				gt.append(y.item())
				pred.append(y_.item())
				
				print(f"ID: {z[0].split('/')[-1].split('.' + fileExt)[0]:<50}\t\t Choronoligal Age: {y.item():.2f}\t\t Predicted Age: {y_.item():.4f}")
				reswriter.writerow({
					'ID': z[0].split('/')[-1].split('.' + fileExt)[0], 
					'Choronoligal Age': y.item(), 
					'Predicted Age': y_.item()
				})
				
				
				
				row_info = z[0].split('/')[-1].split('.' + fileExt)[0]
				row_info = row_info + ',' + str(y.item()) + ',' + str(y_.item()) + ', ,'
				row_info = row_info + ','.join(lastlayer_reshaped.astype(str)) + "\n"
				
				resultsfilelastlayer.write(row_info)
				
				
			gt_arr = np.array(gt)
			pred_arr = np.array(pred)

			for mf in metrics_f.keys():
				if mf in metrics:
					metrics[mf].append( metrics_f[mf](gt_arr, pred_arr) )
				else:
					metrics[mf] = [ metrics_f[mf](gt_arr, pred_arr) ]

			log = f"r2-score = {np.mean(metrics['r2_score']):.4f}"
			log += f" MAE = {np.mean(metrics['mae']):.4f}"

			print("\n" + log)
			
		resultsfilelastlayer.close()
	print("#"*20)


def main():
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("device type: ", device)
	
	model = resnet.generate_model(model_depth=18, n_classes=1, n_input_channels=1, widen_factor=1)
	model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cpu')))
	
	model.avgpool.register_forward_hook(get_features("feats"))
	
	
	dataset_test = get_loader(input_csv, fileExt, batch_size=1, mode="test")
	criterion = nn.MSELoss()
	model.to(device)
	
	start = time.perf_counter_ns()
	
	val_test(model, dataset_test, criterion, device, 1, "test")
	
	end = time.perf_counter_ns()
	
	# print elapsed time in seconds
	print(f"\n\n===> Elapsed time using perf_counter_ns() {(end - start) / (10**9):.2f} seconds | {(end - start) / (60*(10**9)):2f} minutes | {(end - start) / (60*60*(10**9)):2f} hours ")
	


if __name__ == '__main__':
	main()
