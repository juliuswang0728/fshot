import os
import torch
import numpy as np
import pickle

class TransferNetMetrics(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.out_path = os.path.join(cfg.OUTPUT_DIR, 'metrics.pkl')
        self.num_classes = cfg.TRAIN.NUM_CLASSES
        self.accumulated_confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.accumulated_hits = {i: 0. for i in range(self.num_classes)}
        self.num_samples_per_class = {i: 0. for i in range(self.num_classes)}
        self.per_class_accuracy = {i: 0. for i in range(self.num_classes)}
        self.overall_class_accuracy = 0.
        self.mean_class_accuracy = 0.
        self.topk = (1, 5)
        self.accumulated_topk_corrects = {'top{}_acc'.format(k): 0. for k in self.topk}


    def __call__(self, logits, labels):
        return self._get_topk_accuracy(logits, labels)


    def _get_topk_accuracy(self, logits, labels):
        """Computes the precision@k for the specified values of k"""
        maxk = max(self.topk)
        batch_size = labels.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = {}
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            score = correct_k.mul_(100.0 / batch_size)
            res.update({'top{}_acc'.format(k): score})

        return res


    def accumulated_update(self, logits, labels):
        maxk = max(self.topk)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            self.accumulated_topk_corrects['top{}_acc'.format(k)] += correct_k.item()

        predictions = logits.argmax(1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        results = (predictions == labels)
        for hit, pred, label in zip(results, predictions, labels):
            if hit:
                self.accumulated_hits[label] += 1.
            self.num_samples_per_class[label] += 1.
            self.accumulated_confusion_matrix[label, pred] += 1.


    def gather_results(self):
        sum_hits = 0.
        total_samples = 0.
        mean_class_accuracy = 0.
        for k in range(self.num_classes):
            sum_hits += self.accumulated_hits[k]
            total_samples += self.num_samples_per_class[k]
            self.per_class_accuracy[k] = self.accumulated_hits[k] / (self.num_samples_per_class[k] + 1e-8)
            mean_class_accuracy += self.per_class_accuracy[k]

        self.overall_class_accuracy = sum_hits / total_samples
        self.mean_class_accuracy = mean_class_accuracy / self.num_classes

        for k in self.topk:
            score = self.accumulated_topk_corrects['top{}_acc'.format(k)] * 100.0 / total_samples
            self.accumulated_topk_corrects['top{}_acc'.format(k)] = score

        out_dict = {'per_class_accuracies': self.per_class_accuracy,
                    'mean_class_accuracy': self.mean_class_accuracy,
                    'topk': self.accumulated_topk_corrects,
                    'cm': self.accumulated_confusion_matrix,
                    'total_samples': total_samples}

        with open(self.out_path, 'wb') as wp:
            pickle.dump(out_dict, wp, protocol=pickle.HIGHEST_PROTOCOL)


class ProtoNetMetrics(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.topk = (1, )
        self.accumulated_topk_corrects = {'top{}_acc'.format(k): 0. for k in self.topk}
        self.total_num_samples = 0.


    def __call__(self, logits, labels=None):
        return self._get_topk_accuracy(logits, labels)


    def _get_topk_accuracy(self, logits, labels=None):
        """Computes the precision@k for the specified values of k"""
        maxk = max(self.topk)
        k_way = logits.shape[1]
        if labels is None:
            labels = torch.LongTensor(range(0, k_way)).to(logits.device)

        batch_size = labels.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = {}
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            score = correct_k.mul_(100.0 / batch_size)
            res.update({'top{}_acc'.format(k): score})

        return res


    def accumulated_update(self, logits, labels=None):
        maxk = max(self.topk)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        if labels is None:
            labels = torch.LongTensor(range(0, logits.shape[1])).to(logits.device)
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        self.total_num_samples += logits.shape[0]

        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            self.accumulated_topk_corrects['top{}_acc'.format(k)] += correct_k


    def gather_results(self):
        for k in self.topk:
            score = self.accumulated_topk_corrects['top{}_acc'.format(k)] * 100.0 / self.total_num_samples
            self.accumulated_topk_corrects['top{}_acc'.format(k)] = score