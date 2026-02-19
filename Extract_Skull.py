"""
Subtract a brainmask from a full-head MRI (.mgz) to keep only non-brain voxels
(e.g., skull+scalp). Automatically detects non-binary masks (0–255 / 0–1),
binarizes them, and saves both MGZ and NIfTI outputs. Also displays central slices.

Outputs:
  - *_skull_only.mgz / *_skull_only.nii.gz
  - *_nonbrain_mask.mgz / *_nonbrain_mask.nii.gz
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ---------------------- utils ---------------------- #

def load_mgz(path):
	return nib.load(path)

def resample_mask_to_head(mask_img, head_img):
	"""Resample mask to head image grid if needed using nearest-neighbor."""
	from nibabel.processing import resample_from_to
	if mask_img.shape == head_img.shape and np.allclose(mask_img.affine, head_img.affine):
		return mask_img
	return resample_from_to(mask_img, (head_img.shape, head_img.affine), order=0)

def save_both_formats(data, affine, header, out_prefix):
	"""Save as both MGZ and NIfTI."""
	nib.save(nib.freesurfer.mghformat.MGHImage(data, affine, header=header), out_prefix + ".mgz")
	nib.save(nib.Nifti1Image(data, affine), out_prefix + ".nii.gz")

def show_slices(images, titles):
	"""Display the middle axial slice of each image."""
	n = len(images)
	fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
	if n == 1:
		axes = [axes]
	for ax, img, title in zip(axes, images, titles):
		data = img.get_fdata()
		mid = data.shape[2] // 2
		ax.imshow(np.rot90(data[:, :, mid]), cmap="gray")
		ax.set_title(title, fontsize=10)
		ax.axis("off")
	plt.tight_layout()
	plt.show()

# ----------------- mask auto-detection ----------------- #

def _otsu_threshold_0to1(x01):
	"""Compute Otsu threshold on data assumed scaled to [0,1]."""
	x = x01[np.isfinite(x01)]
	if x.size == 0:
		return 0.5
	# 256-bin histogram on [0,1]
	hist, bin_edges = np.histogram(x, bins=256, range=(0.0, 1.0))
	hist = hist.astype(np.float64)
	prob = hist / hist.sum()
	omega = np.cumsum(prob)					 # class probabilities
	mu = np.cumsum(prob * (np.arange(256)))	 # class means (times bin index)
	mu_t = mu[-1]
	# Between-class variance
	sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1.0 - omega) + 1e-12)
	k_star = int(np.nanargmax(sigma_b2))
	# Threshold is the edge between k and k+1
	return (bin_edges[k_star] + bin_edges[k_star + 1]) / 2.0

def auto_binarize_mask(mask_data):
	"""
	Auto-binarize a mask that could be:
	  - already binary {0,1}
	  - {0,255} or general 0–255 uint8
	  - probability-like 0–1
	  - other: use Otsu after scaling to [0,1]
	Returns uint8 array (0/1).
	"""
	m = np.asarray(mask_data)
	m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

	# Fast path: exactly binary?
	uniq = np.unique(m)
	if uniq.size <= 3:  # tolerate small rounding noise
		# Binary-like if only 0 and (near)1 or 0 and 255 (or also all zeros/ones)
		if np.all(np.isin(uniq, [0, 1])) or np.all(np.isin(uniq, [0, 255])):
			print("Case1")
			return (m > 0).astype(np.uint8)

	m_min, m_max = float(m.min()), float(m.max())

	# If it already looks like [0,1]
	if m_max <= 1.5:
		# Probability-like -> Otsu on [0,1]
		thr = _otsu_threshold_0to1(m.astype(np.float32))
		print("Case2")
		return (m > thr).astype(np.uint8)

	# If it looks like uint8 0–255 (or generally ≤ 255)
	if m_max <= 255.0 and np.issubdtype(m.dtype, np.integer):
		x01 = (m / 255.0).astype(np.float32)
		thr = _otsu_threshold_0to1(x01)
		print(f"threshold: {thr}")
		print("Case3")
		return (x01 > thr).astype(np.uint8)

	# General case: scale to [0,1] by min-max and Otsu
	if m_max > m_min:
		x01 = ((m - m_min) / (m_max - m_min)).astype(np.float32)
	else:
		x01 = np.zeros_like(m, dtype=np.float32)
	thr = _otsu_threshold_0to1(x01)
	print("Case4")
	print(f"threshold: {thr}")
	return (x01 > thr).astype(np.uint8)

def maybe_fix_inversion(mask_bin):
	"""
	Heuristic to flip masks that are inverted.
	Brain should roughly occupy ~10–90% of voxels; if outside, try flipping.
	"""
	frac = float(mask_bin.mean())
	if frac < 0.10 or frac > 0.90:
		print("MASK Inversion!!")
		return (1 - mask_bin).astype(np.uint8)
	return mask_bin

# ------------------------- main ------------------------- #

def main(full_head_mgz, brainmask_mgz, out_prefix, no_display=False):
	head_img = load_mgz(full_head_mgz)
	mask_img = load_mgz(brainmask_mgz)

	# Resample mask to match head image grid if needed
	mask_img = resample_mask_to_head(mask_img, head_img)

	head_data = head_img.get_fdata(dtype=np.float32)
	mask_data = mask_img.get_fdata(dtype=np.float32)

	# Auto-binarize & maybe fix inversion
	mask_bin = auto_binarize_mask(mask_data)
	#mask_bin = maybe_fix_inversion(mask_bin)

	# Non-brain mask: 1 where brainmask==0
	nonbrain_mask = (1 - mask_bin).astype(np.uint8)

	# Skull-only intensities (non-brain part of head data)
	skull_only = head_data * nonbrain_mask

	# Save outputs (both formats)
	save_both_formats(skull_only, head_img.affine, head_img.header, f"{out_prefix}_skull_only")
	save_both_formats(nonbrain_mask.astype(np.uint8), head_img.affine, head_img.header, f"{out_prefix}_nonbrain_mask")

	print(f"Saved outputs with prefix '{out_prefix}' in both MGZ and NIfTI formats.")

	# Display central slices (inputs + MGZ outputs)
	if not no_display:
		skull_img = nib.freesurfer.mghformat.MGHImage(skull_only, head_img.affine, header=head_img.header)
		nonbrain_img = nib.freesurfer.mghformat.MGHImage(nonbrain_mask.astype(np.uint8), head_img.affine, header=head_img.header)
		show_slices(
			[head_img, mask_img, skull_img, nonbrain_img],
			["Head MRI Image (input)", "Brain(mask) (input)", "Skull-only (output)", "Non-brain mask (output)"]
		)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Extracting skull using full-head MRI Image (.mgz) and brain Image (.mgz) with auto mask detection.")
	parser.add_argument("--head", required=True, help="Path to head MRI Image .mgz")
	parser.add_argument("--mask", required=True, help="Path to brain mask Image .mgz (any of: binary, 0–255, 0–1)")
	parser.add_argument("--out", default="result", help="Output prefix (default: \"result\")")
	parser.add_argument("--no-display", action="store_true", help="Skip matplotlib display")
	args = parser.parse_args()
	main(args.head, args.mask, args.out, no_display=args.no_display)
