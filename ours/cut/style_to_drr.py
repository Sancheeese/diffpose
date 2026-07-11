import os
import sys
sys.path.append('/home/zsr/project/diffpose/ours/cut/networks')

from .networks import *
import torchvision.transforms.v2 as transforms

class StyleChanger():
    def __init__(self, net_location, device=None, resize=None, opt=None):
        self.net_location = net_location
        self.resize = resize
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        norm_layer = get_norm_layer(norm_type="instance")
        self.net = ResnetGenerator(1, 1, 64, norm_layer=norm_layer, use_dropout=False,
                              no_antialias=False, no_antialias_up=False, n_blocks=9, opt=opt)
        ckpt = torch.load(net_location, map_location=device)
        self.net.load_state_dict(ckpt)
        self.net.to(device=device)
        self.net.eval()
        self.transform = get_transforms(resize)

    def __call__(self, img, reverse=False):
        with torch.no_grad():
            img = to256(img, self.device, reverse=reverse)
            self.transform(img)

            return self.net(img)

def change_style(img, model_location, device, opt=None):
    norm_layer = get_norm_layer(norm_type="instance")
    net = ResnetGenerator(1, 1, 64, norm_layer=norm_layer, use_dropout=False,
                                   no_antialias=False, no_antialias_up=False, n_blocks=9, opt=opt)

    ckpt = torch.load(model_location, map_location=device)
    net.load_state_dict(ckpt)
    net.to(device=device)
    net.eval()

    transforms = get_transforms(resize=256)
    img = transforms(to256(img, device=device))

    return net(img)

def get_transforms(resize=None):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if resize is not None:
        transform_list += [transforms.Resize(resize, antialias=True)]
    transform_list += [transforms.Normalize((0.5,), (0.5,))]
    return transforms.Compose(transform_list)

def to256(img, device, reverse=False):
    img = (img - img.min()) / (img.max() - img.min())
    if reverse:
        img = 1 - img
    # img = torch.tensor(img * 255, device=device, dtype=torch.uint8).to(torch.float32)
    img = (img * 255).clone().detach().to(device=device, dtype=torch.uint8).float()
    return img

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = "/home/zsr/project/contrastive-unpaired-translation/checkpoints/drr_style_nec5"
    change_style(None, os.path.join(root, "latest_net_G.pth"), device, None)

