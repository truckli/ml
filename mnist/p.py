#!/usr/bin/env python
# -*- coding: utf-8 -*-

import struct
from sys import exit

import numpy as np

def show_img(img):
    for i in range(28):
        for j in range(28):
            print("%3d"%img[i*28+j], end = '')
        print("")

def create_x(x_fpath):
    x_buffer = open(x_fpath, "rb").read()
    [nsample, nrow, ncol] = struct.unpack(">LLL", x_buffer[4:16])
    xsize = nrow * ncol
    x = [np.frombuffer(x_buffer[16+xsize*i: 16+(i+1)*xsize], dtype='b') for i in range(nsample)]
    return np.array(x)

def create_y(y_fpath):
    return np.frombuffer(open(y_fpath, "rb").read()[8:], dtype='b')

x_tes = create_x("t10k-images-idx3-ubyte")
y_tes = create_y("t10k-labels-idx1-ubyte")
x_tra = create_x("train-images-idx3-ubyte")
y_tra = create_y("train-labels-idx1-ubyte")


show_img(x_tra[0])
print(y_tra[0])

show_img(x_tes[0])
print(y_tes[0])