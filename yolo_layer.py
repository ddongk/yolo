import torch
import torch.nn as nn


class YOLOLayer(nn.Module):
    def __init__(
        self,
        anchors,
        n_class,
    ):
        super().__init__()