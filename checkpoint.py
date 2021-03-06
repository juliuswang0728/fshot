'''
Most of the contents are adapted from https://github.com/facebookresearch/maskrcnn-benchmark
'''

import os
import torch
import logging
from collections import OrderedDict

def align_and_update_state_dicts(model_state_dict, loaded_state_dict, load_mapping):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    logger = logging.getLogger(__name__)
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    # NOTE: Kaihua Tang, since some modules of current model will be initialized from assigned layer of
    # loaded model, we use load_mapping to do such operation

    mapped_current_keys = current_keys.copy()
    for i, key in enumerate(mapped_current_keys):
        for source_key, target_key in load_mapping.items():
            if source_key in key:
                mapped_current_keys[i] = key.replace(source_key, target_key)
                logger.info("MAPPING {} in current model to {} in loaded model.".format(key, mapped_current_keys[i]))

    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in mapped_current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "REMATCHING! {: <{}} loaded from {: <{}} of shape {}"
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            key = current_keys[idx_new]
            logger.info("NO-MATCHING of current module: {} of shape {}".format(key,
                                    tuple(model_state_dict[key].shape)))
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        # add a control gate for this logger (it's too large)
        if ((not key.startswith('module.'))  and key != key_old) or (key.startswith('module.') and key[7:] != key_old):
            logger.info(
                log_str_template.format(
                    key,
                    max_size,
                    key_old,
                    max_size_loaded,
                    tuple(loaded_state_dict[key_old].shape),
                )
            )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, load_mapping, strict=True):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, load_mapping)

    ignored_keys = []
    for current_key in model.state_dict():
        if current_key in loaded_state_dict:
            if model.state_dict()[current_key].shape != loaded_state_dict[current_key].shape:
                ignored_keys.append(current_key)
                model_state_dict.pop(current_key)
        else:
            ignored_keys.append(current_key)

    model.load_state_dict(model_state_dict, strict=strict)

    return ignored_keys


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        logger=None,
        monitor_unit='epoch',
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.monitor_unit = monitor_unit
        self.current_val_best = 0.
        self.last_saved_at = -1
        self.patience = 0


    def remove_prev_val_best(self):
        if not self.current_val_best:
            return

        try:
            os.remove(self.current_val_best)
            print('removed prev best model {}'.format(self.current_val_best))
        except:
            print('{} not found!'.format(self.current_val_best))
            return


    def record_current_val_best(self, name):
        self.current_val_best = os.path.join(self.save_dir, "{}.pth".format(name))


    def save(self, name, **kwargs):
        if not self.save_dir:
            return
        if self.last_saved_at == kwargs[self.monitor_unit]:
            return

        self.last_saved_at = kwargs[self.monitor_unit]
        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)


    def load_checkpoint(self, path, load_mapping={}, strict=True):
        self.logger.info("Loading checkpoint from {}".format(path))
        self.logger.info("This ONLY loads model parameters and WILL NOT load optimizer/scheduler states")
        if not strict:
            self.logger.info("`strict` is set to False, ignore module parameters not in the checkpoint")
        checkpoint = self._load_file(path)
        ignored_keys = self._load_model(checkpoint, load_mapping, strict)
        self.logger.info("Ignored keys are not loaded: {}".format(ignored_keys))


    def load(self, f=None, with_optim=True, update_schedule=False, load_mapping={}):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint, load_mapping)
        if with_optim:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                if update_schedule:
                    #self.scheduler.last_epoch = checkpoint["iteration"]
                    self.scheduler.last_epoch = checkpoint[self.monitor_unit]
                else:
                    self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        if 'val_best' in checkpoint:
            self.logger.info("previous best validation: {:.4f}".format(checkpoint['val_best']))
        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, load_mapping, strict=True):
        return load_state_dict(self.model, checkpoint.pop("model"), load_mapping, strict)