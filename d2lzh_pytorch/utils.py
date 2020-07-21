import os
import numpy as np
import time
from PIL import Image
import torchvision

from IPython import display
from matplotlib import pyplot as plt

def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols*scale, num_rows*scale)
    _, axes = plt.subplots(num_rows,  num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i*num_cols + j])
            axes[i][i].axes.get_xaxis().set_visible(False)
            axes[i][i].axes.get_yaxis().set_visible(False)

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows*num_cols)]
    show_images(Y, num_rows, num_cols, scale)


if __name__ == "__main__":
    path = '../data/cat.jpeg'
    img = Image.open(path)

    apply(img, torchvision.transforms.RandomHorizontalFlip())


