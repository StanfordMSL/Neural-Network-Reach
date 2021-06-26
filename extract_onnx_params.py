# This code is from https://github.com/verivital/vnn-comp/blob/master/2020/PWL/MIPVerify/scripts/conversion/extract_onnx_params.py
# To run: # python3 extract_onnx_params.py file.onnx

#!/usr/bin/env python3

# import argparse
import os
import sys

import onnx
import onnx.numpy_helper

import numpy as np
import scipy.io as sio

"""
Converts saved `.onnx` files of fully connected networks to `.mat` files in a
format we can process.
"""


def reorder_dims(xs):
    if len(xs.shape) == 4:
        xs = np.transpose(xs, [2, 3, 1, 0])
    if len(xs.shape) == 2:
        xs = np.transpose(xs)
    return xs


def convert_onnx_to_mat(input_path, output_path):
    print("Reading from {} and writing out to {}".format(input_path, output_path))
    model = onnx.load(input_path)
    d = {
        t.name: reorder_dims(onnx.numpy_helper.to_array(t))
        for t in model.graph.initializer
    }

    print("Layers extracted: {}".format(list(d)))
    sio.savemat(output_path, d)


def get_path(relative_path):
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)
    )


if __name__ == "__main__":
    onnxFile = sys.argv[1]
    matFile = onnxFile[:-4] + 'mat'
    convert_onnx_to_mat(onnxFile, matFile)