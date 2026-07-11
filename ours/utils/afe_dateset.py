import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/home/zsr/project/diffpose/ours/utils")
sys.path.append("/home/zsr/project/diffpose/ours")
sys.path.append("/home/zsr/project/diffpose")
from CT_dataset import Transforms
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from ours.cut.style_to_drr import StyleChanger


class AfeDataSet(Dataset):
    def __init__(self, root, net_path, batch_size, device=torch.device('cuda:0')):
        self.root = root
        self.filenames = os.listdir(root)
        self.style_change = StyleChanger(
            net_path,
            device=device,
            resize=256)
        self.device = device
        self.batch_size = batch_size
        self.pos = 0
        self.transforms = Transforms(256)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.filenames[idx]))
        img = transforms.ToTensor()(img).to(torch.float).to(self.device).unsqueeze(0)
        img = self.transforms(img, reverse=False)
        img = self.style_change(img)
        img = self.transforms(img, reverse=False)

        return img

    def get(self):
        imgs = None
        for _ in range(self.batch_size):
            if imgs is None:
                imgs = Image.open(os.path.join(self.root, self.filenames[self.pos]))
                imgs = transforms.ToTensor()(imgs).to(torch.float).to(self.device).unsqueeze(0)
            else:
                img = Image.open(os.path.join(self.root, self.filenames[self.pos]))
                img = transforms.ToTensor()(img).to(torch.float).to(self.device).unsqueeze(0)
                imgs = torch.cat((imgs, img), 0)

            self.pos = (self.pos + 1) % len(self.filenames)

        imgs = self.transforms(imgs, reverse=False)
        imgs = self.style_change(imgs)
        imgs = self.transforms(imgs, reverse=False)
        return imgs

if __name__ == '__main__':
    afe_date = AfeDataSet("/home/zsr/project/diffpose/ours/drrStyle/trainA",
               "/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_solid_nec5/80_net_G.pth",
               4,
               torch.device('cuda:1'))

    for i in range(200):
        img = afe_date.get()
        for im in img:
            plt.figure()
            plt.imshow(im.cpu().squeeze(), cmap="gray")
            plt.show()
