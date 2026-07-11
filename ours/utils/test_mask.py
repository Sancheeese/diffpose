import os

import pydicom
from matplotlib import pyplot as plt

from CT_dataset import IntubationDataset

def get_tube_mask(root, filename):
    seg_filename = os.path.join(root, filename)
    seg_file = pydicom.dcmread(seg_filename)
    seg = seg_file.pixel_array

    return 1 - seg

def get_spine_mask(root, filename):
    names = filename.split(".")
    filename = names[0] + "_spine." + names[1]
    seg_filename = os.path.join(root, filename)
    seg_file = pydicom.dcmread(seg_filename)
    seg = seg_file.pixel_array

    return seg

def get_bone_mask(root, filename):
    names = filename.split(".")
    filename = names[0] + "_bone." + names[1]
    seg_filename = os.path.join(root, filename)
    seg_file = pydicom.dcmread(seg_filename)
    seg = seg_file.pixel_array

    return seg

if __name__ == "__main__":
    root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    x_root = "/home/zsr/project/diffpose/ours/data/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600)

    x_filename = specimen.get_x_filename(3)
    print(x_filename)

    root = "/home/zsr/project/diffpose/ours/seg"
    seg_filename = os.path.join(root, x_filename)
    seg_file = pydicom.dcmread(seg_filename)
    seg = seg_file.pixel_array

    plt.figure()
    plt.imshow(seg)
    plt.show()
