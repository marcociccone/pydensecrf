"""
Adapted from the inference.py to demonstate the usage of the util functions.
"""

import sys
import numpy as np
import pydensecrf.densecrf as dcrf
from dataset_loaders.images.cityscapes import CityscapesDataset as Dataset
from skimage.color import label2rgb
from pydensecrf.utils import (unary_from_labels,
                              create_pairwise_bilateral,
                              create_pairwise_gaussian)

# Get im{read,write} from somewhere.
try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave
    # TODO: Use scipy instead.


# if len(sys.argv) != 4:
#     print("Usage: python {} IMAGE ANNO OUTPUT".format(sys.argv[0]))
#     print("")
#     print("IMAGE and ANNO are inputs and OUTPUT is where the result should be written.")
#     sys.exit(1)

# fn_im = sys.argv[1]
# fn_anno = sys.argv[2]
# fn_output = sys.argv[3]

##################################
### Read images and annotation ###
##################################

# img = imread(fn_im)
# anno_rgb = imread(fn_anno).astype(np.uint32)

n_iter = 20
fn_im = 'valid/f.png'
fn_output = 'valid/f_out.png'
total_img = imread(fn_im)

# Split image and prediction because are concatenated in the same file
img = total_img[:, :2048, :].astype(np.uint8)
anno_rgb = total_img[:, 4096:, :].astype(np.uint8)

# Convert the annotation's RGB color to a label indeces
# NOTE: we disregard the void class
# Everything that is not in cmap is initialized as 0.
# 0 stands for "unknown" or "unsure" as densecrf example do.

cmap = Dataset.get_cmap()[:-1]
cmap = (cmap*255).astype(np.uint8)
anno_lbl = np.zeros(img.shape[:2]).astype(np.uint8)
# Map back from RGB to label representation
for map_key, map_value in enumerate(cmap, start=1):
    mask = np.where((anno_rgb == np.asarray(map_value)).sum(-1) == 3)
    anno_lbl[mask] = map_key

# Compute the number of classes in the label image.
# We subtract one because the number shouldn't include the value 0 which stands
# for "unknown" or "unsure".
labels = anno_lbl
n_labels = len(cmap)

print(n_labels, " labels and \"unknown\" 0: ", set(labels.flat))

###########################
### Setup the CRF model ###
###########################
use_2d = False
# use_2d = True
if use_2d:
    print("Using 2D specialized functions")

    # Example using the DenseCRF2D code
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=True)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
else:
    print("Using generic 2D functions")

    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=True)
    d.setUnaryEnergy(U)

    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(200, 200), schan=(20, 20, 20),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)


####################################
### Do inference and compute MAP ###
####################################

# Run five inference steps.
Q = d.inference(n_iter)

# Find out the most probable class for each pixel.
MAP = np.argmax(Q, axis=0)

# Convert the MAP (labels) back to the corresponding colors and save the image.
MAP = label2rgb(MAP, colors=cmap).astype(np.uint8)
imsave(fn_output, MAP.reshape(img.shape))

# # Just randomly manually run inference iterations
# Q, tmp1, tmp2 = d.startInference()
# for i in range(n_iter):
#     print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
#     d.stepInference(Q, tmp1, tmp2)
