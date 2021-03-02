# CRF-model-for-segmentation
A supervisingly easy way to conduct CRF for post processing your initial segmentation results

You may use this method combined with: Nvidia-semantic-segmentation/UNet/HRNet/PSPNet etc.

# The input you need:
 - initial segmentation results (RGB labels)
 - original input images

# The output 
 - a new segmentation mask

# Setup
  the only package you need is pydensecrf, you may consider install this with:
  - 'pip install cython'
  - 'pip install git+https://github.com/lucasb-eyer/pydensecrf.git'
# Fine-tuning/Changes you need to do
- the parameter you need to fine tune: rgb_image_test.py, line 14: gt_prob. 
- The changes needed: rgb_image_test.py, line 45: set the num_classes, line 56-57: change paths. 
