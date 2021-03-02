#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#peter LIN, 2021.3.1
"""

"""
import pydensecrf.densecrf as dense_crf
import cv2
from cv2 import imread
import matplotlib.pyplot as plt
from densecrf2 import crf_model, potentials
import numpy as np
# Create unary potential
unary = potentials.UnaryPotentialFromProbabilities(gt_prob=0.7)

bilateral_pairwise = potentials.BilateralPotential(
    sdims=80,
    schan=13,
    compatibility=4,
    kernel=dense_crf.DIAG_KERNEL,
    normalization=dense_crf.NORMALIZE_SYMMETRIC
)

gaussian_pairwise = potentials.GaussianPotential(
    sigma=3, 
    compatibility=2,
    kernel=dense_crf.DIAG_KERNEL,
    normalization=dense_crf.NORMALIZE_SYMMETRIC
)

# =============================================================================
# Create CRF model and add potentials
# =============================================================================
#zero_unsure:  whether zero is a class, if its False, it means zero canb be any of other classes
# =============================================================================
# crf = crf_model.DenseCRF(
#     num_classes = 3,
#     zero_unsure = True,              # The number of output classes
#     unary_potential=unary,
#     pairwise_potentials=[bilateral_pairwise, gaussian_pairwise],
#     use_2d = 'rgb-2d'                #'rgb-1d' or 'rgb-2d' or 'non-rgb'
# )
# =============================================================================
crf = crf_model.DenseCRF(
    num_classes = 7,
    zero_unsure = False,              # The number of output classes
    unary_potential=unary,
    pairwise_potentials=[bilateral_pairwise, gaussian_pairwise],
    use_2d = 'rgb-2d'                #'rgb-1d' or 'rgb-2d' or 'non-rgb'
)


# =============================================================================
# Load image and probabilities
# =============================================================================
image = imread('/home/jianzhe/CRF-semantic-segmentation/best_images/_2eb24BRlq6tpw-IqYm6KQ_left_input.png')
probabilities = imread('/home/jianzhe/CRF-semantic-segmentation/best_images/_2eb24BRlq6tpw-IqYm6KQ_left_prediction.png')

# [h, w, c] = colors.shape
# probabilities = np.zeros((h, w, 1))
# for i in range(0, h):
#      for j in range(0, w):
#          if (colors[i,j,:] == [255, 97, 39]).all():
#              probabilities[i,j,0] = 1
#          elif (colors[i,j,:] == [244, 35,232]).all():
#              probabilities[i,j,0] = 2    
#          elif (colors[i,j,:] == [102,102,153]).all():
#              probabilities[i,j,0] = 3 
#          elif (colors[i,j,:] == [85, 85, 85]).all():
#              probabilities[i,j,0] = 4 
#          elif (colors[i,j,:] == [204,153,102]).all():
#              probabilities[i,j,0] = 5
#          elif (colors[i,j,:] == [128, 64,128]).all():
#              probabilities[i,j,0] = 6
#          elif (colors[i,j,:] == [0,  0,  0]).all():
#              probabilities[i,j,0] = 7  
#          else:
#              break
# =============================================================================
# Set the CRF model
# =============================================================================
#label_source: whether label is from softmax, or other type of label.
crf.set_image(
    image=image,
    probabilities=probabilities,
    colour_axis=-1,                  # The axis corresponding to colour in the image numpy shape
    class_axis=-1,                   # The axis corresponding to which class in the probabilities shape
    label_source = 'label'           # where the label come from, 'softmax' or 'label'
)

# =============================================================================
# run the inference
# =============================================================================
crf.perform_inference(20)  # The CRF model will restart run.
new_mask80 = crf.segmentation_map
print(crf.kl_divergence)
cv2.imwrite("/home/jianzhe/CRF-semantic-segmentation/mask.jpg", new_mask80)












