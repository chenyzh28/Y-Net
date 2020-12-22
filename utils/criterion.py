import torch.nn as nn
import torch
from torch.nn import functional as F
from .loss import OhemCrossEntropy2d

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        return loss1 + loss2 * 0.4

class CriterionExtDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, reduce=True, se_weight=0.2):
        super(CriterionExtDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        self.criterion2 = torch.nn.BCELoss()
        self.se_weight = se_weight
        if not reduce:
            print("disabled the reduce.")


    def forward(self, preds, target, target_cls):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        loss2 = self.criterion2(torch.sigmoid(preds[1]), target_cls)

        return loss1 + self.se_weight * loss2

class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2*0.4