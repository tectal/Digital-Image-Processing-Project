#coding:utf-8  
# -*- coding: utf-8 -*-  
# @Time    : 2018/5/21
# @Author  : yangguofeng  
# @File    : cub_util.py  
# @Software: Sublime Test 3

import os
import numpy as np
from scipy.misc import imread, imresize

class CUB200(object):
    def __init__(self, path, image_size=(224, 224)):
        self._path = path
        self._size = image_size

    def _classes(self):
        return os.listdir(self._path)

    def _load_image(self, category, im_name):
        return imresize(imread(os.path.join(self._path, category, im_name), mode="RGB"), self._size)

    def load_dataset(self, num_per_class=None):
        classes = self._classes()
        all_images = []
        all_labels = []
        for c in classes:
            class_images = os.listdir(os.path.join(self._path, c))
            if num_per_class is not None:
                class_images = np.random.choice(class_images, num_per_class)
            for image_name in class_images:
                all_images.append(self._load_image(c, image_name))
                all_labels.append(c)
        return np.array(all_images).astype(float), np.array(all_labels)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    DATA_DIR = os.path.expanduser(os.path.join("DATA_DIR", "CUB_200_2011"))
    CUB_DIR = os.path.join(DATA_DIR, "CUB_200_2011", "images")
    X, lbl = CUB200(CUB_DIR).load_dataset()
    n = X.shape[0]
    rnd_birds = np.vstack([np.hstack([X[np.random.choice(n)] for i in range(20)])
                           for j in range(10)])
    plt.figure(figsize=(6, 6))
    plt.imshow(rnd_birds / 255)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.title("CUB_200_2011 200 Birds", fontsize=30)
    plt.show()