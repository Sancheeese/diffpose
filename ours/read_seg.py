import nrrd
import torch
from matplotlib import pyplot as plt

data, header = nrrd.read("/home/zsr/project/diffpose/ours/bone_seg/notag/金建功_93909427_20240515_1_24.nrrd")
data = torch.from_numpy(data).float()
data = data.squeeze()
data = data.permute(1, 0)

plt.figure()
plt.imshow(data)
plt.show()
