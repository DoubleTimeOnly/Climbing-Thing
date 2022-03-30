from typing import Set, List, Union


class PerformanceMetrics:
    def __init__(self, truth: Union[Set, List], prediction: Union[Set, List]):
        if isinstance(truth, List):
            truth = set(truth)
        if isinstance(prediction, List):
            prediction = set(prediction)

        self.true_positives = truth.intersection(prediction)
        self.false_negatives = truth.difference(prediction)
        self.false_positives = prediction.difference(truth)

        num_tp = len(self.true_positives)
        num_fn = len(self.false_negatives)
        num_fp = len(self.false_positives)

        self.precision = num_tp / (num_tp + num_fp)
        self.recall = num_tp / (num_tp + num_fn)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

    def __str__(self):
        output = ""
        output += f"F1: {self.f1} | Precision: {self.precision} | Recall: {self.recall}"
        output += f" | TP: {len(self.true_positives)} {self.true_positives} " \
                  f"| FP: {len(self.false_positives)} {self.false_positives} " \
                  f"| FN: {len(self.false_negatives)} {self.false_negatives}"
        return output
