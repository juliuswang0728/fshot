# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.MODEL = CN()

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.HEIGHT = 600
_C.INPUT.WIDTH = 400
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.DEBUG = False

_C.DATALOADER.FP_DATASET = CN()  # fashion product dataset
# C.DATALOADER.FP_DATASET.PATH = './data/fashion-product-images'
_C.DATALOADER.FP_DATASET.ROOTDIR = './data/fashion-product-images-small'
_C.DATALOADER.FP_DATASET.TOPK = 20
_C.DATALOADER.FP_DATASET.BOTTOMK = -1
_C.DATALOADER.FP_DATASET.VAL_DATA_FRACTION = 0.2
# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.ARCHITECTURE = "TransferNet"
_C.MODEL.CONV_BODY = CN()
_C.MODEL.CONV_BODY.ARCH = "res50"
_C.MODEL.CONV_BODY.FREEZE_NUM_BLOCKS = 2
_C.MODEL.CONV_BODY.RESNET50_FIRST_BLOCK_INDEX = 4
_C.MODEL.CONV_BODY.RESNET50_FEAT_DIM = 2048
_C.MODEL.NORM_FEATURES = False
_C.MODEL.NORM_PROTOTYPES = False
_C.MODEL.RADIUS_PROTOTYPES = 1.

_C.MODEL.PROTONET = CN()
_C.MODEL.PROTONET.VIS_EMBEDDING_DIM = 1024
_C.MODEL.PROTONET.DISTANCE_METRIC = "Euclidean"  # "Euclidean" / "Cosine"
_C.MODEL.PROTONET.COSINE_RADIUS = 1.
# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.SHUFFLE = True
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.LR_BASE = 0.1
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0001
_C.TRAIN.LR_DECAY = 0.5
_C.TRAIN.LR_DECAY_EPOCH = 1
_C.TRAIN.LR_DECAY_EPISODE = 50000
_C.TRAIN.NUM_CLASSES = 20
_C.TRAIN.K_WAY = 15      # number of classes per episode
_C.TRAIN.N_SHOT = 1     # number of support examples per class (in an episode)
_C.TRAIN.MAX_EPOCH = 5
_C.TRAIN.MAX_EPISODE = 10000
_C.TRAIN.SAVE_CKPT_EPOCH = 1
_C.TRAIN.SAVE_CKPT_EPISODE = 20000
_C.TRAIN.PATIENCE = 3  # early terminate after `TRAIN.PATIENCE` consecutive validation scores achieve no better
_C.TRAIN.PRINT_PERIOD = 100  # in number of batches
_C.TRAIN.VAL_EPOCH = 1
_C.TRAIN.VAL_BATCH_SIZE = 64
# ---------------------------------------------------------------------------- #
# Loss options
# ---------------------------------------------------------------------------- #
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.USE_LABEL_SMOOTHING = False
_C.TRAIN.LOSS.LABEL_SMOOTHING_EPS = 0.05
# ---------------------------------------------------------------------------- #

# Test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.K_WAY = 15      # number of classes per episode (can be different from TRAIN.K_WAY by one's design choice)
_C.TEST.N_SHOT = 1     # number of support examples per class (though, can be different from TRAIN.K_SHOT, but suggested to be similar in ProtoNet work)
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./checkpoints"
_C.GLOVE_DIR = "."
