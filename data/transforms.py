import torch
import torchvision.transforms.functional as F
from PIL import Image


class ComposeD(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class NormalizeD(object):
    def __init__(self, mean=None, std=None):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

        if mean is not None: self.mean = mean
        if std is not None: self.std = std

    def __call__(self, x):
        img, annot = x['img'], x['annot']
        return {'img': F.normalize(img, mean, std), 'annot': annot}


class ToTensorD(object):
    def __init__(self):
        pass

    def __call__(self, x):
        # x['img'] should be pil image
        img, annot = x['img'], x['annot']
        return {'img': F.to_tensor(img), 'annot': torch.from_numpy(annot)}


class ResizeD(object):
    def __init__(self, size: tuple):
        self.h, self.w = size

    def __call__(self, x):
        img, annot = x['img'], x['annot']
        w, h = img.size
        # print(type(img))
        img = F.resize(img, (self.h, self.w), Image.BILINEAR)
        # print(w, h, self.w, self.h)
        w_ratio = self.w / w
        h_ratio = self.h / h

        annot[:, 0] = annot[:, 0] * w_ratio
        annot[:, 1] = annot[:, 1] * h_ratio
        annot[:, 2] = annot[:, 2] * w_ratio
        annot[:, 3] = annot[:, 3] * h_ratio
        return {'img': img, 'annot': annot}
