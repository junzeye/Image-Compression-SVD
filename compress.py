import numpy as np
from skimage import io, img_as_float
import matplotlib.pyplot as plt
import argparse, os
from svd import svd_compress

# parse input arguments args
parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=float, default=0.2,
                    help = "ratio of compressed image size vs original size")
parser.add_argument('--fname', type=str, help = "pathname to the image file")
parser.add_argument('--as_gray', type=bool, choices=[True, False])
args = parser.parse_args()

img = io.imread(args.fname, as_gray = args.as_gray)
im2arr = img_as_float(img) # dimensions: height x width x channel

m, n, channels_cnt, U_C, Sigma_C, V_C = svd_compress(im2arr, args.ratio)

def save_array(U: list[np.ndarray], S: list[np.ndarray], V: list[np.ndarray]):
    os.makedirs("compressed_data", exist_ok = True) # create directory to store compressed data
    # save compressed data as pickled numpy array
    with open('compressed_data/U.npz') as f1:
        np.savez(*U)
    with open('compressed_data/Sigma.npz') as f2:
        np.savez(*S)
    with open('compressed_data/V.npz') as f3:
        np.savez(*V)

save_array(U_C, Sigma_C, V_C)

A_C = [None] * channels_cnt
for i in range(channels_cnt):
    A_C[i] = U_C[i] @ np.diag(Sigma_C[i]) @ V_C[i].T

color_matrices = np.zeros((m, n, channels_cnt))
for i in range(channels_cnt):
    color_matrices[:,:,i] = A_C[i]

if channels_cnt == 4 or channels_cnt == 2: # if there is an alpha channel
    alphas = np.clip(color_matrices[:,:,-1], 0, 1)
    plt.imshow(X = np.clip(color_matrices[:,:,:-1], 0 , 1), alpha = alphas)
else:
    plt.imshow(np.clip(color_matrices, 0 , 1))

# plt.imsave(fname = 'out.png', arr = np.clip(color_matrices, 0, 1), format = 'png') # to save image, use this.