import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from networks.loss import TransferNetLoss
from networks.metric import TransferNetMetrics
from networks.utils import layer_init
from networks.base import BaseNet


class TransferNet(BaseNet):
    def __init__(self, cfg, logger=None, random_state=1234):
        super(TransferNet, self).__init__(cfg, logger, random_state)

        self.class_scores = nn.Linear(self.out_dim, cfg.TRAIN.NUM_CLASSES)
        self._init_modules()


    def _init_modules(self):
        if self.cfg.MODEL.CONV_BODY.ARCH == 'res50':
            first_block = self.cfg.MODEL.CONV_BODY.RESNET50_FIRST_BLOCK_INDEX

            for i, layer in enumerate(self.feature_extractor):
                if i < first_block + self.cfg.MODEL.CONV_BODY.FREEZE_NUM_BLOCKS:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            raise NotImplementedError

        layer_init(self.class_scores)


    def forward(self, inputs, labels):
        feats = self.feature_extractor(inputs)
        feats = feats.squeeze(2).squeeze(2)
        class_scores = self.class_scores(feats)

        return class_scores


class FinetuneNet(BaseNet):
    def __init__(self, cfg, logger=None, random_state=1234):
        super(FinetuneNet, self).__init__(cfg, logger, random_state)

        if self.cfg.MODEL.NORM_PROTOTYPES:
            self.class_scores_w = torch.nn.Parameter(torch.FloatTensor(self.out_dim, cfg.TRAIN.NUM_CLASSES))
            self.class_scores = self._class_scores
            torch.nn.init.xavier_normal_(self.class_scores_w)
        else:
            self.class_scores = nn.Linear(self.out_dim, cfg.TRAIN.NUM_CLASSES)
            layer_init(self.class_scores)

        self._init_modules()


    def _init_modules(self):
        if self.cfg.MODEL.CONV_BODY.ARCH == 'res50':
            first_block = self.cfg.MODEL.CONV_BODY.RESNET50_FIRST_BLOCK_INDEX

            for i, layer in enumerate(self.feature_extractor):
                if i < first_block + self.cfg.MODEL.CONV_BODY.FREEZE_NUM_BLOCKS:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            raise NotImplementedError


    def _w_dot_x(self, x, w):
        if self.cfg.MODEL.NORM_FEATURES:
            normed_x = F.normalize(x, p=2, dim=1)
        if self.cfg.MODEL.NORM_PROTOTYPES:
            normed_w = F.normalize(w, p=2, dim=0)

        return self.cfg.MODEL.RADIUS_PROTOTYPES * torch.matmul(normed_x, normed_w)


    def _class_scores(self, x):
        return self._w_dot_x(x, self.class_scores_w)


    def forward(self, inputs, labels):
        feats = self.feature_extractor(inputs)
        feats = feats.squeeze(2).squeeze(2)
        class_scores = self.class_scores(feats)

        return class_scores



