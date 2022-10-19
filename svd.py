import numpy as np
from PIL import Image
import argparse

im = Image.open('1.jpg')
im2arr = np.array(im) # im2arr.shape: height x width x channel

# convert back to image
# arr2im = Image.fromarray(im2arr)
