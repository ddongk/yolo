# https://github.com/ayooshkathuria/pytorch-yolo-v3/blob/master/darknet.py
# https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/tool/darknet2pytorch.py
# https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py

from .parser import parse_cfg

import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/fe45fb64ba4c21130643f9b2569da7bb667356b3/tool/darknet2pytorch.py#L10
class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


#for shortcut, route
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


def create_modules(blocks, net_info):
    '''
    blocks : output of parse_cfg func. cfg파일에서 파싱한 block들, [net] 제외.
    return : nn.ModuleList() class
    '''
    module_list = nn.ModuleList()
    out_filters = [int(net_info['channels'])]  # 모든 레이어의 아웃풋 채널 수

    for index, block in enumerate(blocks):
        module = nn.Sequential()
        type_ = block['type']

        #convolution layer
        if type_ == 'convolutional':
            activation = block['activation']
            bn = int(
                block['batch_normalize'])  # whether to use batchnorm or not
            filters = int(block['filters'])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            is_pad = int(block["pad"])
            pad = (kernel_size - 1) // 2 if is_pad else 0  # for 'same' padding
            module.add_module(
                f'conv_{index}',
                nn.Conv2d(out_filters[-1],
                          filters,
                          kernel_size,
                          stride,
                          pad,
                          bias=not bn))
            if bn:
                module.add_module(
                    f'batch_norm_{index}',
                    nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))

            if activation == 'leaky':
                module.add_module(f'leaky_{index}',
                                  nn.LeakyReLU(0.1, inplace=False))
            elif activation == 'mish':
                module.add_module(f'mish_{index}', Mish())
            elif activation == 'linear':
                pass
            else:
                print(f'unknown activation function {activation} at {index}')

        #shortcut layer
        elif type_ == 'shortcut':
            filters = out_filters[int(block["from"])]
            module.add_module(f"shortcut_{index}", EmptyLayer())
            # print(f"shortcut_{index}", filters, int(block["from"]))

        #route layer
        elif type_ == 'route':
            layers = list(map(int,
                              block['layers'].split(',')))  # split 하고 모두 int로
            # layers = [i if i > 0 else i + index for i in layers]  # i가 음수일경우 양수로 변경 ->굳이 필요 x
            filters = sum([out_filters[i] for i in layers])
            module.add_module(f"route_{index}", EmptyLayer())
            # print(f"route_{index}", [out_filters[i] for i in layers], layers)

        #upsample layer
        elif type_ == "upsample":
            #https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe -> yolo는 nearest 사용
            # stride = int(x["stride"]) # stride는 다 2라서 걍 주석
            # expand를 쓰기도 하는데 ....이유는 모르겠다. 맨위 링크 참조
            # tensor.expand랑 nearest랑 같은 연산인지 확인 해봐야 할 듯
            # upsample = nn.Upsample(scale_factor=2, mode="nearest")
            # module.add_module(f"upsample_{index}", upsample)
            upsample = Upsample(scale_factor=int(block["stride"]),
                                mode="nearest")
            module.add_module(f"upsample_{index}", upsample)

        # max pool layer
        elif type_ == "maxpool":
            stride = int(block['stride'])
            size = int(block['size'])
            if stride == 1 and size % 2:
                pad = size // 2
                maxpool = nn.MaxPool2d(size, stride, pad)
            else:
                print(f"unknown maxpool condition at {index}")
            module.add_module(f'maxpool_{index}', maxpool)
        #yolo layer
        elif type_ == "yolo":
            mask = list(map(int, block["mask"].split(",")))

            anchors = list(map(int, block["anchors"].split(",")))
            anchors = [(anchors[i], anchors[i + 1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            num_classes = int(block['classes'])

            img_size = int(net_info['height'])

            yolo_layer = DarknetYOLOLayer(anchors, img_size, num_classes)
            module.add_module(f"yolo_layer_{index}", yolo_layer)

        else:
            print(f'unknwon layer type : {type_}, {index}th')
        module_list.append(module)
        out_filters.append(filters)
    return module_list


class DarknetYOLOLayer(nn.Module):
    '''
    for detection
    '''
    def __init__(self, anchors, img_size, num_classes):
        super().__init__()

        self.anchors = anchors
        self.na = len(anchors)
        self.nc = num_classes
        self.img_size = img_size
        self.ng = 0

    def compute_grid_offsets(self, ng, cuda=True):
        self.ng = ng
        g = self.ng
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.ns = self.img_size / self.ng
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g,
                                                         g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g,
                                             1).t().view([1, 1, g,
                                                          g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.ns, a_h / self.ns)
                                           for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.na, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.na, 1, 1))

    # https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py
    # 여기서 퍼옴 아주 깔끔한 코드임 good! -> 기존꺼랑 차이는 없음 결과값에서
    def forward(self, x, targets=None):
        # x => batch, (num_classes+5)*3, H, W

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        nb = x.size(0)  # number of batch
        ng = x.size(2)  # grid size

        # (nb, na*no, ng, ng) --> (nb, na, ng, ng, no)
        prediction = x.view(nb, self.na, self.nc+5, ng, ng)\
                            .permute(0, 1, 3, 4,2).contiguous()

        # If grid size does not match current we compute new offsets
        if ng != self.ng:
            self.compute_grid_offsets(ng, cuda=x.is_cuda)

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(nb, -1, 4) * self.ns,
                pred_conf.view(nb, -1, 1),
                pred_cls.view(nb, -1, self.nc),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            return output


class Darknet(nn.Module):
    def __init__(self, cfg):
        super(Darknet, self).__init__()

        self.net_info, self.blocks = parse_cfg(cfg)
        self.module_list = create_modules(self.blocks, self.net_info)

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x, targets=None):
        yolo_out = []
        outputs = []
        # print(x.shape)
        for i, (block, module) in enumerate(zip(self.blocks,
                                                self.module_list)):
            type_ = block["type"]
            loss = 0
            if type_ in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)

            # fowarding route layer
            elif type_ == "route":
                x = torch.cat([outputs[int(layer_i)] \
                        for layer_i in block["layers"].split(",")], 1)

            # forwarding shortcut layer
            elif type_ == 'shortcut':
                x = outputs[-1] + \
                    outputs[int(block["from"])]  # 바로 이전과 from의 sum

            # forwarding yolo layer
            elif type_ == "yolo":
                print(module)
                x, yolo_loss = module[0](x, targets)
                loss += yolo_loss
                yolo_out.append(x)
            outputs.append(x)
        yolo_outputs = torch.cat(yolo_out, 1)
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_weights(self, weight_file):
        # Open the weights file
        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        with open(weight_file, "rb") as f:
            import numpy as np
            header = np.fromfile(f, dtype=np.int32,
                                 count=5)  # First five are header values
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        for i, (block, module) in enumerate(zip(self.blocks,
                                                self.module_list)):
            if block["type"] == "convolutional":
                conv_layer = module[0]
                if block["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(
                    conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    # test 필요 아직 안돌려본 코드
    def save_darknet_weights(self, path, cutoff=-1):
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (block, module) in enumerate(
                zip(self.blocks[:cutoff], self.module_list[:cutoff])):

            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

    #     fp.close()


if __name__ == "__main__":
    pass