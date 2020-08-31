from .box import *
import torchvision.ops as ops
import torch


# batch 단위로 필요
def post_processing(x, thres_conf=0.3, thres_nms=0.5):
    result = []
    boxes = torch.empty(0, 4)
    scores = []
    labels = []
    for i, pred in enumerate(x):
        # print(pred.shape) #torch.Size([10647, 85])
        pred = pred[pred[:, 4] > thres_conf]
        if pred.size(0) == 0:
            # print(pred.shape)
            break
        # print(pred[:, :4])
        pred = cxcywh2xyxy(pred)
        # print(pred[:, :4])
        cls_score, cls_idx = torch.max(pred[:, 5:], 1, keepdim=True)
        keep = ops.boxes.batched_nms(pred[:, :4], cls_score[:, 0],
                                     cls_idx[:, 0], thres_nms)
        boxes = pred[keep][:, :4]
        scores = cls_score[keep]
        labels = cls_idx[keep]

    return boxes, scores, labels
