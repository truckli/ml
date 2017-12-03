#!/usr/bin/env python
# -*- coding: utf-8 -*-

import struct
from sys import exit

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA


def show_img(img):
    for i in range(8):
        for j in range(8):
            print("%3d"%img[i*8+j], end = '')
        print("")

def create_x(x_fpath):
    df = pd.read_csv(x_fpath)
    return df 

def create_y(y_fpath):
    return np.frombuffer(open(y_fpath, "rb").read()[8:], dtype='b')


df_tra = pd.read_csv("optdigits.tra",header=None)
X_tra = df_tra.loc[:,:63]
y_tra = df_tra[64]
df_tes = pd.read_csv("optdigits.tes",header=None)
X_tes = df_tes.loc[:,:63]
y_tes = df_tes[64]

clf = LinearSVC(random_state=0)
clf.fit(X_tra, y_tra)
(clf.score(X_tes, y_tes))
