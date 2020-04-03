import torch
import torchvision.models as models
import torch.nn as nn
from networks.loss import TransferNetLoss, ProtoNetLoss
from networks.metric import TransferNetMetrics, ProtoNetMetrics

LossEvaluators = {'TransferNet': TransferNetLoss, 'FinetuneNet': TransferNetLoss,
                  'ProtoNet': ProtoNetLoss, 'RelNet': ProtoNetLoss}

MetricEvaluators = {'TransferNet': TransferNetMetrics, 'FinetuneNet': TransferNetMetrics,
                  'ProtoNet': ProtoNetMetrics, 'RelNet': ProtoNetMetrics}

class BaseNet(nn.Module):
    def __init__(self, cfg, logger=None, random_state=1234):
        super(BaseNet, self).__init__()
        self.cfg = cfg
        self.logger = logger
        self.random_state = random_state

        self.feature_extractor, self.out_dim = self.get_feature_extractor()
        self.loss_evaluator = get_loss_evaluator(cfg.MODEL.ARCHITECTURE)(cfg)
        self.metric_evaluator = get_metric_evaluator(cfg.MODEL.ARCHITECTURE)(cfg)


    def get_feature_extractor(self):
        if self.cfg.MODEL.CONV_BODY.ARCH == 'res50':
            conv_body = models.resnet50(pretrained=True)
            out_dim = conv_body.fc.in_features

            feature_extractor = list(conv_body.children())[:-1]

            return nn.Sequential(*feature_extractor), out_dim

        else:
            raise NotImplementedError


    def forward(self, inputs, labels=None):
        pass


def get_loss_evaluator(architecture):
    return LossEvaluators[architecture]


def get_metric_evaluator(architecture):
    return MetricEvaluators[architecture]
