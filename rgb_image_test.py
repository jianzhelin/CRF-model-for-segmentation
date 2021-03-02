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
image = imread('path/to/your/image/.png')
probabilities = imread('path/to/your/rgb/label/.png')

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
cv2.imwrite("path/to/save/the/new/mask.jpg", new_mask80)












