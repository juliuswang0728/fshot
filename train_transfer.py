import os
import time
import datetime
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader

import torch.optim as optim

from logger import setup_logger, MetricLogger
from dataset import FashionProductImages
from configs import cfg
from configs.utils import save_config

from networks.transfer_net import TransferNet
from networks.metric import TransferNetMetrics
from checkpoint import Checkpointer


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def do_eval(cfg, model, dataloader, logger, task):
    val_metrics = TransferNetMetrics(cfg)

    model.eval()
    num_images = 0
    for iteration, data in enumerate(dataloader):
        inputs, targets = data['image'].to(device), data['label_articleType'].to(device)
        cls_scores = model(inputs, targets)
        val_metrics.accumulated_update(cls_scores, targets)
        num_images += len(targets)

    val_metrics.gather_results()
    logger.info('num of images: {}'.format(num_images))
    logger.info('{} overall acc: {:.4f}, mean class acc: {:.4f}'.format(task,
                                                                        val_metrics.overall_class_accuracy,
                                                                        val_metrics.mean_class_accuracy))

    return val_metrics



def do_train(cfg, model, train_dataloader, val_dataloader, logger):
    # define optimizer
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR_BASE,
                              momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise NotImplementedError

    # define learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, cfg.TRAIN.LR_DECAY)
    checkpointer = Checkpointer(model, optimizer, lr_scheduler, cfg.OUTPUT_DIR, logger)

    training_args = {}
    #training_args['iteration'] = 1
    training_args['epoch'] = 1
    training_args['val_best'] = 0.
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load()
        training_args.update(extra_checkpoint_data)

    #start_iter = training_args['iteration']
    start_epoch = training_args['epoch']
    checkpointer.current_val_best = training_args['val_best']

    meters = MetricLogger(delimiter="  ")
    end = time.time()
    start_training_time = time.time()

    for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH + 1):
        training_args['epoch'] = epoch
        model.train()
        for inner_iter, data in enumerate(train_dataloader):
            #training_args['iteration'] = iteration
            data_time = time.time() - end
            inputs, targets = data['image'].to(device), data['label_articleType'].to(device)
            cls_scores = model(inputs, targets)
            losses = model.loss_evaluator(cls_scores, targets)
            metrics = model.metric_evaluator(cls_scores, targets)

            total_loss = sum(loss for loss in losses.values())
            meters.update(loss=total_loss, **losses, **metrics)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if inner_iter % cfg.TRAIN.PRINT_PERIOD == 0:
                eta_seconds = meters.time.global_avg * (len(train_dataloader) * cfg.TRAIN.MAX_EPOCH -
                                                        (epoch - 1) * len(train_dataloader) - inner_iter)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {ep}/{max_ep} (iter: {iter}/{max_iter})",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        ep=epoch,
                        max_ep=cfg.TRAIN.MAX_EPOCH,
                        iter=inner_iter,
                        max_iter=len(train_dataloader),
                        meters=str(meters),
                        lr=optimizer.param_groups[-1]["lr"],
                        memory=(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) if torch.cuda.is_available() else 0.,
                    )
                )

        if epoch % cfg.TRAIN.VAL_EPOCH == 0:
            logger.info('start evaluating at epoch {}'.format(epoch))
            val_metrics = do_eval(cfg, model, val_dataloader, logger, 'validation')
            if val_metrics.mean_class_accuracy > checkpointer.current_val_best:
                checkpointer.current_val_best = val_metrics.mean_class_accuracy
                training_args['val_best'] = checkpointer.current_val_best
                checkpointer.save("model_{:04d}_val_{:.4f}".format(epoch, checkpointer.current_val_best), **training_args)
                checkpointer.patience = 0
            else:
                checkpointer.patience += 1

            logger.info('current patience: {}/{}'.format(checkpointer.patience, cfg.TRAIN.PATIENCE))

        if epoch == cfg.TRAIN.MAX_EPOCH or epoch % cfg.TRAIN.SAVE_CKPT_EPOCH == 0 or checkpointer.patience == cfg.TRAIN.PATIENCE:
            checkpointer.save("model_{:04d}".format(epoch), **training_args)

        if checkpointer.patience == cfg.TRAIN.PATIENCE:
            logger.info('Max patience triggered. Early terminate training')
            break

        if epoch % cfg.TRAIN.LR_DECAY_EPOCH == 0:
            logger.info("lr decayed to {:.4f}".format(optimizer.param_groups[-1]["lr"]))
            lr_scheduler.step()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))

    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (epoch - start_epoch if epoch > start_epoch else 1)
        )
    )

def main():
    parser = argparse.ArgumentParser(description="PyTorch Transfer Learning Task")
    parser.add_argument("--config", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    np.random.seed(1234)
    torch.manual_seed(1234)
    if 'cuda' in device:
        torch.cuda.manual_seed_all(1234)

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

    # create disjoint train / val / test sets
    train_dataset = FashionProductImages(cfg=cfg, split='train', logger=logger)
    val_dataset = FashionProductImages(cfg=cfg, split='val', logger=logger)
    val_dataset.params['shuffle'] = False
    val_dataset.params['batch_size'] = cfg.TRAIN.VAL_BATCH_SIZE
    test_dataset = FashionProductImages(cfg=cfg, split='test', logger=logger)
    test_dataset.params['shuffle'] = False

    train_dataloader = DataLoader(train_dataset, **train_dataset.params)
    val_dataloader = DataLoader(val_dataset, **val_dataset.params)
    test_dataloader = DataLoader(test_dataset, **test_dataset.params)

    model = TransferNet(cfg, cfg.TRAIN.NUM_CLASSES, logger)
    model.to(device)

    do_train(cfg, model, train_dataloader, val_dataloader, logger)
    do_eval(cfg, model, test_dataloader, logger, 'test')

if __name__ == "__main__":
    main()
