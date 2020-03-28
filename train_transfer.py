import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader

from logger import setup_logger
from dataset import FashionProductImages
from configs import cfg
from configs.utils import save_config

def do_val(dataloader):
    pass


def main():
    parser = argparse.ArgumentParser(description="PyTorch Transfer Learning Task")
    parser.add_argument("--config", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT_DIR):
        print('creating output dir: {}'.format(cfg.OUTPUT_DIR))
        os.makedirs(cfg.OUTPUT_DIR)

    logger = setup_logger("transfer_pretrain", cfg.OUTPUT_DIR)
    logger.info("Loaded configuration file {}".format(args.config))
    with open(args.config, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    # disjoint train / val / test sets
    train_dataset = FashionProductImages(cfg=cfg, split='train', logger=logger)
    val_dataset = FashionProductImages(cfg=cfg, split='val', logger=logger)
    test_dataset = FashionProductImages(cfg=cfg, split='test', logger=logger)

    train_dataloader = DataLoader(train_dataset, **train_dataset.params)
    val_dataloader = DataLoader(val_dataset, **val_dataset.params)
    test_dataloader = DataLoader(test_dataset, **test_dataset.params)

    for step, data in enumerate(train_dataloader):
        inputs, targets = data['image'], data['label_articleType']
        break


if __name__ == "__main__":
    main()
