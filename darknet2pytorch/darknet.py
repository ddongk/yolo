# https://github.com/ayooshkathuria/pytorch-yolo-v3/blob/master/darknet.py
# https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/tool/darknet2pytorch.py
from .parser import parse_cfg

import torch
import torch.nn as nn

import numpy as np


# feature transform for detection : https://taeu.github.io/paper/deeplearning-paper-yolov3/
def create_grids(na, ng):
    gx = torch.arange(ng).repeat(1, ng).reshape(ng, ng)
    gy = gx.clone().T

    m = na * ng * ng
    grid = torch.stack((gx, gy), 2).repeat(3, 1, 1).reshape(m, 2)
    return grid


class DarknetYOLOLayer(nn.Module):
    '''
    for detection
    '''
    def __init__(self, anchors, in_dimH, in_dimW, num_classes):

        super().__init__()

        self.num_anchors = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.num_classes = num_classes
        self.in_dimW = in_dimW
        self.in_dimH = in_dimH

    def forward(self, x):
        # x => B, 255, H, W
        '''
        정방 행렬만 가능 -> 나중에 수정 예정
        cuda tensor만 가능 -> 나중에 수정 예정
        '''
        stride_H = self.in_dimH // x.size(2)
        stride_W = self.in_dimW // x.size(3)
        assert (stride_H == stride_W)

        # 각 변수들
        nb = x.shape[0]  # number of batch
        na = self.num_anchors  # number of anchors
        no = self.num_classes + 5  # 5 : x,y,w,h,conf
        ng = self.in_dimH // stride_H  # grid size
        m = na * ng * ng
        ns = stride_H
        # print(nb, na, no, ng, m, ns, self.anchors)

        CxCy = create_grids(na, ng).cuda()
        PwPy = ((self.anchors / ns).view(na, 1, 2).repeat(1, ng * ng,
                                                          1).view(m,
                                                                  2)).cuda()

        # (bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)
        x = x.view(nb, na, no, ng, ng).permute(0, 1, 3, 4, 2).contiguous()
        # (bs, 3, 13, 13, 85) --> (bs, 3*13*13, 85)
        x = x.view(nb, m, no)

        # slicing 좀 상태 안좋음 메모리 문제 때문에, -> 아래 링크 참조
        # 그래서 아래 슬라이싱 주석처리 해놓음
        # https://discuss.pytorch.org/t/unable-to-convert-pytorch-model-to-onnx/64158
        # x[:, :, :2] = torch.sigmoid(x[:, :, :2]) + self.CxCy
        xy = torch.sigmoid(x[:, :, :2]) + CxCy  #x,y

        # x[:, :, 2:4] = (torch.exp(x[:, :, 2:4]) * self.PwPy ) * self.ns
        wh = torch.exp(x[:, :, 2:4]) * PwPy  # w,h

        # x[:, :, 4] = torch.sigmoid(x[:, :, 4])
        confidence = torch.sigmoid(x[:, :, 4]).unsqueeze(2)  # confidence score

        # 논문에서 softmax 안씀, 각클래스는 독립적이라고 여기고
        # x[:, :, 5:self.no] = torch.sigmoid(x[:, :, 5:self.no])
        pred_cls = torch.sigmoid(x[:, :, 5:no])  # class

        # confidence에 class prediction 곱해야하는지 말아야하는지 헷갈.. -> 안하는것같다.. ?
        # return torch.cat((xy * ns, wh * ns, confidence, pred_cls), 2)
        return torch.cat((xy * ns, wh * ns, confidence, pred_cls * confidence),
                         2)


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

            num_classes = int(block['classes'])
            in_dimH = int(net_info['height'])
            in_dimW = int(net_info['width'])

            yololayer = DarknetYOLOLayer(anchors, in_dimH, in_dimW,
                                         num_classes)
            module.add_module(f"YOLOlayer_{index}", yololayer)

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
        self.module_list = create_modules(self.blocks, self.net_info)

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

        self.in_dimH = int(self.net_info['height'])  # H
        self.in_dimW = int(self.net_info['width'])  # W

    def forward(self, x):
        yolo_out = []
        outputs = {}
        # print(x.shape)

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
                x = self.module_list[index](x)
                yolo_out.append(x)
        return torch.cat(yolo_out, 1)

    def load_weights(self, weight_file):
        def load_conv_bn(weights, ptr, conv_model, bn_model):
            #Get the number of weights of Batch Norm Layer
            num_bn_biases = bn_model.bias.numel()

            #Load the weights
            bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
            ptr += num_bn_biases
            bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
            ptr += num_bn_biases
            bn_running_mean = torch.from_numpy(weights[ptr:ptr +
                                                       num_bn_biases])
            ptr += num_bn_biases
            bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
            ptr += num_bn_biases

            #Cast the loaded weights into dims of model weights.
            bn_biases = bn_biases.view_as(bn_model.bias.data)
            bn_weights = bn_weights.view_as(bn_model.weight.data)
            bn_running_mean = bn_running_mean.view_as(bn_model.running_mean)
            bn_running_var = bn_running_var.view_as(bn_model.running_var)

            #Copy the data to model
            bn_model.bias.data.copy_(bn_biases)
            bn_model.weight.data.copy_(bn_weights)
            bn_model.running_mean.copy_(bn_running_mean)
            bn_model.running_var.copy_(bn_running_var)

            #BN이 있으므로 conv의 bias는 필요 없음.
            #Let us load the weights for the Convolutional layers
            num_weights = conv_model.weight.numel()

            #Do the same as above for weights
            conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
            ptr = ptr + num_weights

            conv_weights = conv_weights.view_as(conv_model.weight.data)
            conv_model.weight.data.copy_(conv_weights)
            return ptr

        def load_conv(weights, ptr, conv_model):
            #bn이 없으므로 conv에 bias도 load해야함
            #Number of biases
            num_biases = conv_model.bias.numel()

            #Load the weights
            conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
            ptr = ptr + num_biases

            #reshape the loaded weights according to the dims of the model weights
            conv_biases = conv_biases.view_as(conv_model.bias.data)

            #Finally copy the data
            conv_model.bias.data.copy_(conv_biases)

            #Let us load the weights for the Convolutional layers
            num_weights = conv_model.weight.numel()

            #Do the same as above for weights
            conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
            ptr = ptr + num_weights

            conv_weights = conv_weights.view_as(conv_model.weight.data)
            conv_model.weight.data.copy_(conv_weights)
            return ptr

        # Open the weights file
        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        with open(weight_file, "rb") as f:
            header = np.fromfile(f, dtype=np.int32,
                                 count=5)  # First five are header values
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        for i in range(len(self.module_list)):
            type_ = self.blocks[i]["type"]
            # print(i, type_)

            if type_ == "convolutional":
                model = self.module_list[i]
                bn = int(self.blocks[i]['batch_normalize'])
                if bn: ptr = load_conv_bn(weights, ptr, model[0], model[1])
                else: ptr = load_conv(weights, ptr, model[0])
                # print(ptr, weights.size)
        print("done")

    def save_weights(self):
        pass


if __name__ == "__main__":
    pass
