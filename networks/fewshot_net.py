import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from networks.loss import TransferNetLoss
from networks.metric import TransferNetMetrics
from networks.utils import layer_init
from networks.base import BaseNet


class ProtoNet(BaseNet):
    def __init__(self, cfg, logger=None, random_state=1234):
        super(ProtoNet, self).__init__(cfg, logger, random_state)
        self.k_way = self.cfg.TRAIN.K_WAY
        self.n_shot = self.cfg.TRAIN.N_SHOT
        self.num_samples_one_episode = self.k_way * self.n_shot + self.k_way

        self.distance_metric = self._get_distance_metric()
        self.vis_embedding = nn.Linear(cfg.MODEL.CONV_BODY.RESNET50_FEAT_DIM, cfg.MODEL.PROTONET.VIS_EMBEDDING_DIM)
        self.xent = nn.CrossEntropyLoss()
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

        layer_init(self.vis_embedding)


    def _get_distance_metric(self):
        def _euclidean(x, y):
            norm = torch.pow(x - y, 2).sum(dim=1)
            return -norm

        def _cosine(x, y):
            norm_x = F.normalize(x, p=2, dim=1)
            norm_y = F.normalize(y, p=2, dim=1)
            return self.cfg.MODEL.PROTONET.COSINE_RADIUS * (norm_x * norm_y).sum(dim=1)

        if self.cfg.MODEL.PROTONET.DISTANCE_METRIC == 'Euclidean':
            return _euclidean
        elif self.cfg.MODEL.PROTONET.DISTANCE_METRIC == 'Cosine':
            return _cosine
        else:
            raise NotImplementedError


    def forward(self, inputs, labels=None):
        feats = self.feature_extractor(inputs)
        feats = feats.squeeze(2).squeeze(2)
        feats = self.vis_embedding(feats)

        support_feats = feats[0:self.k_way * self.n_shot]
        support_feats = support_feats.split(self.n_shot)
        query_feats = feats[self.k_way * self.n_shot::]

        feat_protos = torch.cat([feat.mean(dim=0).view(1, -1) for feat in support_feats])
        # (1, 1, 1, 2, 2, 2, 3, 3, 3, ...)
        query_rep = torch.cat([torch.cat([x.view(1, -1)] * self.k_way) for x in query_feats])
        # (1, 2, 3, 1, 2, 3, 1, 2, 3, ...)
        num_queries = query_feats.shape[0]
        protos_rep = torch.cat([feat_protos] * num_queries)
        # cross_dot(query, protos)
        cross_dot_qp_matrix = self.distance_metric(query_rep, protos_rep).reshape(num_queries, self.k_way)

        return cross_dot_qp_matrix


class RelNet(BaseNet):
    def __init__(self, cfg, logger=None, random_state=1234):
        super(RelNet, self).__init__(cfg, logger, random_state)
        self.k_way = self.cfg.TRAIN.K_WAY
        self.n_shot = self.cfg.TRAIN.N_SHOT
        self.num_samples_one_episode = self.k_way * self.n_shot + self.k_way

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


    def forward(self, inputs, labels):
        feats = self.feature_extractor(inputs)
        feats = feats.squeeze(2).squeeze(2)