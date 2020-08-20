# https://github.com/ayooshkathuria/pytorch-yolo-v3/blob/master/darknet.py
# https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/tool/darknet2pytorch.py
from .parser import parse_cfg

import torch
import torch.nn as nn


#for shortcut, route
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


def create_modules(blocks):
    '''
    blocks : output of parse_cfg func. cfg파일에서 파싱한 block들, [net] 제외.
    return : nn.ModuleList() class
    '''

    module_list = nn.ModuleList()

    prev_filters = 3  # 이전 레이어 채널 수, 초기는 3임
    out_filters = []  # 모든 레이어의 아웃풋 채널 수

    for index, block in enumerate(blocks):

        module = nn.Sequential()

        type_ = block['type']

        #convolution layer
        if type_ == 'convolutional':
            activation = block['activation']
            batch_norm = int(
                block['batch_normalize'])  # whether to use batchnorm or not
            filters = int(block['filters'])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            is_pad = int(block["pad"])
            pad = (kernel_size - 1) // 2 if is_pad else 0  # for 'same' padding

            if batch_norm:
                module.add_module(
                    f'conv{index}',
                    nn.Conv2d(prev_filters,
                              filters,
                              kernel_size,
                              stride,
                              pad,
                              bias=False))
                module.add_module(f'bn{index}', nn.BatchNorm2d(filters))

            else:
                module.add_module(
                    f'conv{index}',
                    nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))

            if activation == 'leaky':
                module.add_module(f'leaky{index}',
                                  nn.LeakyReLU(0.1, inplace=True))
            elif activation == 'relu':
                module.add_module(f'relu{index}', nn.ReLU(inplace=True))

        #shortcut layer
        elif type_ == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{index}", shortcut)

        #route layer
        elif type_ == 'route':
            route = EmptyLayer()
            module.add_module(f"route_{index}", route)

            layers = list(map(int,
                              block['layers'].split(',')))  # split 하고 모두 int로
            layers = [i if i > 0 else i + index
                      for i in layers]  # i가 음수일경우 양수로 변경

            n_layers = len(layers)  # route 할 layer 개수

            if n_layers == 1:  # 이 case랑 shortcut은 분명히 다름 -> 이전 채널 수가 다름
                filters = out_filters[layers[0]]
            elif n_layers == 2:
                filters = out_filters[layers[0]] + out_filters[layers[1]]
            # elif n_layers == 4:

        #upsample layer
        elif type_ == "upsample":
            #https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe
            # yolo는 nearest 사용
            # stride = int(x["stride"]) # stride는 다 2라서 걍 주석
            # expand를 쓰기도 하는데 ....이유는 모르겠다. 맨위 링크 참조
            # tensor.expand랑 nearest랑 같은 연산인지 확인 해봐야 할 듯
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module(f"upsample_{index}", upsample)

        #yolo layer
        elif type_ == "yolo":
            mask = list(map(int, block["mask"].split(",")))
            anchors = list(map(int, block["anchors"].split(",")))
            anchors = [(anchors[i], anchors[i + 1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

        module_list.append(module)
        prev_filters = filters
        out_filters.append(filters)
    return module_list


class Darknet(nn.Module):
    def __init__(self, cfg, inference=False):
        super(Darknet, self).__init__()
        self.inference = inference
        self.train_ = not self.inference

        self.net_info, self.blocks = parse_cfg(cfg)
        self.module_list = create_modules(self.blocks)

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x, CUDA=True):
        yolo_out = []
        outputs = {}

        for index, block in enumerate(self.blocks):
            type_ = block["type"]

            if type_ in ['convolutional', 'upsample']:
                x = self.module_list[index](x)
                outputs[index] = x

            # fowarding route layer
            elif type_ == "route":
                layers = list(map(
                    int, block['layers'].split(',')))  # split 하고 모두 int로
                layers = [i if i > 0 else i + index
                          for i in layers]  # i가 음수일경우 양수로 변경

                n_layers = len(layers)  # route 할 layer 개수

                if n_layers == 1:  # 이 case랑 shortcut은 분명히 다름 -> 이전 채널 수가 다름
                    x = outputs[layers[0]]
                    outputs[index] = x

                elif n_layers == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[index] = x

                # elif n_layers == 4:

            # forwarding shortcut layer
            elif type_ == 'shortcut':
                from_ = int(block["from"])
                from_ = from_ if from_ > 0 else from_ + index

                x = outputs[index - 1] + outputs[from_]  # 바로 이전과 from의 sum

                activation = self.blocks[index]['activation']
                # if activation == "leaky": # 나중에 추가
                outputs[index] = x

            elif type_ == "yolo":
                yolo_out.append(x)

        return yolo_out


if __name__ == "__main__":
    pass