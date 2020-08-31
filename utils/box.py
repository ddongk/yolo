import torch
import numpy as np


def xyxy2xywh(boxes):
    '''
    boxes : (n, 4) or (n, >4), torch tensor
    X Y X Y -> X Y W H
    '''
    # if isinstance(boxes, np.ndarray):
    #     boxes = torch.from_numpy(boxes)
    new_boxes = boxes.clone()

    new_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
    new_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # h

    return new_boxes


def xywh2xyxy(boxes):
    '''
    boxes : (n, 4) or (n, >4), torch tensor
    X Y W H -> X Y X Y
    '''
    # if isinstance(boxes, np.ndarray):
    #     boxes = torch.from_numpy(boxes)
    new_boxes = boxes.clone()

    new_boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    new_boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    return new_boxes


def cxcywh2xyxy(boxes):
    '''
    boxes : (n, 4) or (n, >4), torch tensor
    Cx Cy W H -> X Y X Y
    '''
    # if isinstance(boxes, np.ndarray):
    #     boxes = torch.from_numpy(boxes)
    new_boxes = boxes.clone()
    w_half = boxes[:, 2] / 2.0
    h_half = boxes[:, 3] / 2.0

    new_boxes[:, 0] = boxes[:, 0] - w_half
    new_boxes[:, 1] = boxes[:, 1] - h_half
    new_boxes[:, 2] = boxes[:, 0] + w_half
    new_boxes[:, 3] = boxes[:, 1] + h_half

    return new_boxes


def xyxy2cxcywh(boxes):
    '''
    boxes : (n, 4) or (n, >4), torch tensor
    X Y X Y -> Cx Cy W H
    '''
    # if isinstance(boxes, np.ndarray):
    #     boxes = torch.from_numpy(boxes)
    new_boxes = boxes.clone()

    w_ = boxes[:, 2] - boxes[:, 0]
    h_ = boxes[:, 3] - boxes[:, 1]

    new_boxes[:, 0] = boxes[:, 0] + (w_ / 2.0)
    new_boxes[:, 1] = boxes[:, 1] + (h_ / 2.0)
    new_boxes[:, 2] = w_
    new_boxes[:, 3] = h_

    return new_boxes


def xywh2cxcywh(boxes):
    '''
    boxes : (n, 4) or (n, >4), torch tensor
    X Y W H -> Cx Cy W H
    '''
    # if isinstance(boxes, np.ndarray):
    #     boxes = torch.from_numpy(boxes)
    new_boxes = boxes.clone()

    new_boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2.0
    new_boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2.0
    return new_boxes


def cxcywh2xywh(boxes):
    '''
    boxes : (n, 4) or (n, >4), torch tensor
    Cx Cy W H -> X Y W H
    '''
    # if isinstance(boxes, np.ndarray):
    #     boxes = torch.from_numpy(boxes)
    new_boxes = boxes.clone()
    new_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    new_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    return new_boxes