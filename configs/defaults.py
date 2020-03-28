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

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.DEBUG = False

_C.DATALOADER.FP_DATASET = CN()     # fashion product dataset
#C.DATALOADER.FP_DATASET.PATH = './data/fashion-product-images'
_C.DATALOADER.FP_DATASET.ROOTDIR = './data/fashion-product-images-small'
_C.DATALOADER.FP_DATASET.TOPK = 20
_C.DATALOADER.FP_DATASET.BOTTOMK = -1
_C.DATALOADER.FP_DATASET.VAL_DATA_FRACTION = 0.2
# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.CONV_BODY = "res50"
# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.SHUFFLE = True
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.SAVE_CKPT_PERIOD = 2000
# ---------------------------------------------------------------------------- #
# Test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./checkpoints"
_C.GLOVE_DIR = "."
