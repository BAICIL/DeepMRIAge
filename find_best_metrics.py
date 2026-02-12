"""Age prediction Model
   Author: Jay Shah
   File: find_best_metrics.py, finds best validation model
   based on defined metrics; here MAE
   
   Contact: jgshah1@asu.edu
   Web: https://www.public.asu.edu/~jgshah1/
"""

import glob

logs = sorted(glob.glob("*.out"))

for log in logs:
	with open(log, 'r') as logfile:
		lines = logfile.readlines()
		best_mae, best_r2 = (0, 100000), (0, 0)
		for line in lines:
			if 'Working on' in line:
				foldname = line.split(' ')[-1].split('\n')[0]
			if 'Testing on TEST Dataset at' in line:
				epoch = line.split(' ')[-10]
				mae = float(line.split(' ')[-1])
				if mae < best_mae[1]:
					best_mae = (epoch, mae)
				r2 = float(line.split(' ')[-4])
				if r2 > best_r2[1]:
					best_r2 = (epoch, r2)
				
		print(f"Fold:{foldname}, MAE:{best_mae}, R2:{best_r2}")
				