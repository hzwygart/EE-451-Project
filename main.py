import os
import numpy as np
import pandas as pd
from PIL import Image
from Segmentation import segment_image

#Paths
TEST_IMAGE_DIR  = 'test_images/'
TRAIN_IMAGE_DIR = 'train_images/'
OUTPUT_CSV      = 'submission.csv'

img = np.array(Image.open(os.path.join(TRAIN_IMAGE_DIR, 'L1000849.jpg'))) # Load a sample image for testing
mask = segment_image(img)