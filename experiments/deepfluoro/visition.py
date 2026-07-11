from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

models = sorted(Path("checkpoints/").glob("specimen_01_epoch*.ckpt"))
losses = []
for model_name in models:
    ckpt = torch.load(model_name, weights_only=True)
    loss = ckpt["loss"]
    epoch = ckpt["epoch"]
    losses.append((loss, epoch))

losses.sort(key=lambda x: x[-1])
print(losses)

losses = losses[:21]
y = []
for loss, _ in losses:
    y.append(loss)

x = np.arange(0, 1001, 50)

plt.plot(x, y)
plt.title("subject_01")
plt.show()


