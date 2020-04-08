import os
import time
import datetime
import numpy as np
import torch
import argparse
import importlib
from torch.utils.data import DataLoader

import torch.optim as optim

from logger import setup_logger, MetricLogger
from dataset import FashionProductImagesFewShot
from configs import cfg
from configs.utils import save_config

from networks.fewshot_net import ProtoNet, RelNet
from networks.metric import TransferNetMetrics
from checkpoint import Checkpointer

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

architectures = {'ProtoNet': ProtoNet, 'RelNet': RelNet}


def do_test(cfg, model, dataloader, logger, task, load_ckpt):
    if load_ckpt is not None:
        checkpointer = Checkpointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger, monitor_unit='episode')
        checkpointer.load_checkpoint(load_ckpt)

    val_metrics = model.metric_evaluator

    model.eval()
    num_images = 0
    meters = MetricLogger(delimiter="  ")

    logger.info('Start testing...')
    start_testing_time = time.time()
    end = time.time()
    for iteration, data in enumerate(dataloader):
        data_time = time.time() - end
        inputs, labels = torch.cat(data['images']).to(device), data['labels'].to(device)
        logits = model(inputs)
        val_metrics.accumulated_update(logits, labels)
        num_images += logits.shape[0]

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (len(dataloader) - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 50 == 0 and iteration > 0:
            logger.info('eta: {}, iter: {}/{}'.format(eta_string, iteration, len(dataloader)))

    val_metrics.gather_results()
    logger.info('num of images: {}'.format(num_images))
    logger.info('{} top1 acc: {:.4f}'.format(task, val_metrics.accumulated_topk_corrects['top1_acc']))

    total = time.time() - start_testing_time
    total_time_str = str(datetime.timedelta(seconds=total))

    logger.info("Total testing time: {}".format(total_time_str))
    return val_metrics


def do_train(cfg, model, train_dataloader, logger, load_ckpt=None):
    # define optimizer
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR_BASE,
                              momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise NotImplementedError

    # define learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, cfg.TRAIN.LR_DECAY)
    checkpointer = Checkpointer(model, optimizer, lr_scheduler, cfg.OUTPUT_DIR, logger, monitor_unit='episode')

    training_args = {}
    # training_args['iteration'] = 1
    training_args['episode'] = 1
    if load_ckpt:
        checkpointer.load_checkpoint(load_ckpt, strict=False)

    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load()
        training_args.update(extra_checkpoint_data)

    start_episode = training_args['episode']
    episode = training_args['episode']

    meters = MetricLogger(delimiter="  ")
    end = time.time()
    start_training_time = time.time()

    model.train()
    break_while = False
    while not break_while:
        for inner_iter, data in enumerate(train_dataloader):
            training_args['episode'] = episode
            data_time = time.time() - end
            # targets = torch.cat(data['labels']).to(device)
            inputs = torch.cat(data['images']).to(device)
            logits = model(inputs)
            losses = model.loss_evaluator(logits)
            metrics = model.metric_evaluator(logits)

            total_loss = sum(loss for loss in losses.values())
            meters.update(loss=total_loss, **losses, **metrics)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if inner_iter % cfg.TRAIN.PRINT_PERIOD == 0:
                eta_seconds = meters.time.global_avg * (cfg.TRAIN.MAX_EPISODE - episode)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "episode: {ep}/{max_ep}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        ep=episode,
                        max_ep=cfg.TRAIN.MAX_EPISODE,
                        iter=inner_iter,
                        max_iter=len(train_dataloader),
                        meters=str(meters),
                        lr=optimizer.param_groups[-1]["lr"],
                        memory=(
                                torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) if torch.cuda.is_available() else 0.,
                    )
                )

            if episode % cfg.TRAIN.LR_DECAY_EPISODE == 0:
                logger.info("lr decayed to {:.4f}".format(optimizer.param_groups[-1]["lr"]))
                lr_scheduler.step()

            if episode == cfg.TRAIN.MAX_EPISODE:
                break_while = True
                checkpointer.save("model_{:06d}".format(episode), **training_args)
                break

            if episode % cfg.TRAIN.SAVE_CKPT_EPISODE == 0:
                checkpointer.save("model_{:06d}".format(episode), **training_args)

            episode += 1

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))

    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (episode - start_episode if episode > start_episode else 1)
        )
    )

def main():
    parser = argparse.ArgumentParser(description="PyTorch Transfer Learning Task")
    parser.add_argument("--config", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--load_ckpt", default=None, metavar="FILE", help="path to the pretrained model to load ",
                        type=str)
    parser.add_argument('--test_only', action='store_true',
                        help="test the model (need to provide model path to --load_ckpt)")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.test_only:
        assert os.path.exists(args.load_ckpt), "You need to provide the valid model path using --load_ckpt"

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

    logger = setup_logger("fewshot", cfg.OUTPUT_DIR)
    logger.info("Loaded configuration file {}".format(args.config))
    with open(args.config, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = architectures[cfg.MODEL.ARCHITECTURE](cfg, logger)
    model.to(device)

    meta_test_dataset = FashionProductImagesFewShot(cfg=cfg, split='test',
                                                    k_way=cfg.TEST.K_WAY, n_shot=cfg.TEST.N_SHOT, logger=logger)
    meta_test_dataset.params['shuffle'] = False
    meta_test_dataset.params['batch_size'] = 1
    meta_test_dataloader = DataLoader(meta_test_dataset, **meta_test_dataset.params)

    if args.test_only:
        do_test(cfg, model, meta_test_dataloader, logger, 'test', args.load_ckpt)
    else:
        meta_train_dataset = FashionProductImagesFewShot(cfg=cfg, split='train',
                                                         k_way=cfg.TRAIN.K_WAY, n_shot=cfg.TRAIN.N_SHOT, logger=logger)
        meta_train_dataloader = DataLoader(meta_train_dataset, **meta_train_dataset.params)

        do_train(cfg, model, meta_train_dataloader, logger, args.load_ckpt)
        do_test(cfg, model, meta_test_dataloader, logger, 'test', None)


    '''for inner_iter, data in enumerate(meta_test_dataloader):
        if inner_iter % 250 == 0 and inner_iter > 0:
            print('[test] {}/{}'.format(inner_iter, len(meta_test_dataloader)))
        if inner_iter == len(meta_test_dataset) - 1:
            print('[test] {}/{}'.format(inner_iter, len(meta_test_dataloader)))'''


if __name__ == "__main__":
    main()