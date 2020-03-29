import torch


class TransferNetMetrics(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_classes = cfg.TRAIN.NUM_CLASSES
        self.accumulated_hits = {i: 0. for i in range(self.num_classes)}
        self.num_samples_per_class = {i: 0. for i in range(self.num_classes)}
        self.per_class_accuracy = {i: 0. for i in range(self.num_classes)}
        self.overall_class_accuracy = 0.
        self.mean_class_accuracy = 0.


    def __call__(self, logits, labels):
        cls_acc = self._get_cls_accuracy(logits, labels)
        out_metrics = {'cls_acc': cls_acc}

        return out_metrics


    def _get_cls_accuracy(self, logits, labels):
        accuracy = (logits.argmax(1) == labels).float().mean().item()

        return accuracy


    def accumulated_update(self, logits, labels):
        results = logits.argmax(1) == labels
        for hit, label in zip(results, labels):
            label = label.item()
            if hit:
                self.accumulated_hits[label] += 1.
            self.num_samples_per_class[label] += 1.


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

