import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from networks.loss import TransferNetLoss
from networks.metric import TransferNetMetrics
from networks.utils import layer_init

class TransferNet(nn.Module):
    def __init__(self, cfg, num_classes, logger=None, random_state=1234):
        super(TransferNet, self).__init__()

        self.cfg = cfg
        self.logger = logger
        self.random_state = random_state

        self.feature_extractor, out_dim = self.get_feature_extractor()
        self.class_scores = nn.Linear(out_dim, num_classes)
        self.loss_evaluator = TransferNetLoss(cfg)
        self.metric_evaluator = TransferNetMetrics(cfg)

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


    def get_feature_extractor(self):
        if self.cfg.MODEL.CONV_BODY.ARCH == 'res50':
            conv_body = models.resnet50(pretrained=True)
            out_dim = conv_body.fc.in_features

            feature_extractor = list(conv_body.children())[:-1]

            return nn.Sequential(*feature_extractor), out_dim

        else:
            raise NotImplementedError


    def forward(self, inputs, labels):
        feats = self.feature_extractor(inputs)
        feats = feats.squeeze(2).squeeze(2)
        class_scores = self.class_scores(feats)
        #class_prob = F.softmax(class_scores)

        return class_scores


