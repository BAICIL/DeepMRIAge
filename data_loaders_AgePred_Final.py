from functools import partial
from glob import glob
from torch.utils import data
from monai import transforms as T
import nibabel as nib
import numpy as np
import os
import random, csv
import torch
import sys



import numpy as np
import nibabel as nib
from nibabel.orientations import (
    io_orientation, axcodes2ornt, ornt_transform,
    apply_orientation, inv_ornt_aff
)

def is_decimal_string(s: str) -> bool:
    """
    Returns True if s is a valid non-negative decimal number using only digits and
    at most one dot. Accepts: "12", "12.45", ".45", "12."
    Rejects: "", ".", "12..45", "12a", "1.2.3"
    """
    if not isinstance(s, str) or not s:
        return False

    dot_count = 0
    digit_count = 0

    for ch in s:
        if ch.isdigit():
            digit_count += 1
        elif ch == ".":
            dot_count += 1
            if dot_count > 1:
                return False
        else:
            return False

    # must contain at least one digit (so "." is not allowed)
    return digit_count > 0

def load_mgz_as_ras_array2(mgz_path: str, dtype=np.float32):
	
	img = nib.load(mgz_path)
	#data = img.get_fdata(dtype=np.float32)
	data = img.get_fdata()

	cur = io_orientation(img.affine)
	tgt = axcodes2ornt(("R","A","S"))      # choose your target axcodes here
	xfm = ornt_transform(cur, tgt)

	new_data = apply_orientation(data, xfm)
	#new_aff  = img.affine @ inv_ornt_aff(xfm, img.shape)

	#out = nib.MGHImage(new_data, new_aff, header=img.header)
	#out.update_header()
	#nib.save(out, "scan_RAS.mgz")
	
	return new_data



def load_mgz_as_ras_array(mgz_path: str, dtype=np.float32):
    img = nib.load(mgz_path)

    # Reorient to closest canonical orientation (RAS+ in nibabel)
    img_ras = nib.as_closest_canonical(img)

    # Data array in RAS+ axis order
    vol = img_ras.get_fdata(dtype=dtype)  # applies scaling, returns float array
    #vol = img_ras.get_fdata()  # applies scaling, returns float array

    #return vol, img_ras.affine
    return vol



class DataFolder(data.Dataset):
	def __init__(self, csvFile, image_type, transform, mode='train'):
		self.__image_reader = {
			'np': lambda url: np.load(url),
			'nii': lambda url: nib.load(url).get_fdata(),
			'nii.gz': lambda url: nib.load(url).get_fdata(),
			'mgz': lambda url: load_mgz_as_ras_array(url),
		}
		
		self.__supported_extensions = self.__image_reader.keys()
		assert image_type in self.__supported_extensions
		assert transform != None

		self.csvFile = csvFile		
		self.image_type = image_type
		self.transform = transform
		self.mode = mode
		self.data_urls = []
		self.data_labels = []
		self.data_index = []
		self.num_classes = 0

		self.__process()

		print(f"Total data loaded = {len(self.data_urls)}")

	def __process(self):
		
		#print(self.mode)
		
		## Reads csv file with image path and ages
		with open(self.csvFile,'r') as infile:
			reader = csv.reader(infile)
			age_data = dict((rows[0],rows[1]) for rows in reader)
			#print(age_data)
		
		for path in list(age_data.keys()):
			age = age_data[path]
			
			#if not age.isdigit():
			if not is_decimal_string(age):
				del age_data[path]
				continue
							
			self.data_urls += [path]
			self.data_labels += [float(age)]
	
		
		self.data_index = list(range(len(self)))
		
		#print(self.data_urls)
		#print(self.data_labels)
		
		assert len(self) > 0

	def __read(self, url):
		#print(f"URL in READ f: {url}")
		return self.__image_reader[self.image_type](url)

	def __getitem__(self, index):
		img = self.__read(self.data_urls[self.data_index[index]])
		lbl = self.data_labels[self.data_index[index]]

		max_intensity = 255
		img -= np.min(img)
		img /= max_intensity
		
		
		return torch.FloatTensor(img).unsqueeze(0), torch.LongTensor([lbl]), self.data_urls[self.data_index[index]]

	def __len__(self):
		return len(self.data_urls)


def get_loader(csvFile, fileExt, crop_size=101, image_size=101, 
			batch_size=5, dataset='PTH_HC', mode='train', num_workers=1):
	"""Build and return a data loader."""
	transform = []

	if mode == 'train':
		transform.append(T.RandGaussianNoise())
		transform.append(T.RandShiftIntensity(30))
		transform.append(T.RandStdShiftIntensity(3))
		transform.append(T.RandBiasField())
		transform.append(T.RandScaleIntensity(0.25))
		transform.append(T.RandAdjustContrast())
		transform.append(T.RandGaussianSmooth())
		transform.append(T.RandGaussianSharpen())
		transform.append(T.RandHistogramShift())
		# transform.append(T.RandGibbsNoise())
		# transform.append(T.RandKSpaceSpikeNoise())
		transform.append(T.RandRotate())
		transform.append(T.RandFlip())

	
	transform.append(T.ToTensor())
	transform = T.Compose(transform)

	
	dataset = DataFolder(csvFile, fileExt, transform, mode)
	
		

	data_loader = data.DataLoader(dataset=dataset,
								batch_size=batch_size,
								shuffle=(mode=='train'),
								drop_last=True,
								num_workers=num_workers)
								
								
	

	
	return data_loader


