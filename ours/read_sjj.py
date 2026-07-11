import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('sjj_data.h5', 'r') as f:
    img = f["images/images/img_1_1"][0]
    img1 = np.squeeze(img)
    plt.figure()
    plt.imshow(img1, cmap="gray")
    plt.show()

