import unittest
from climbing_thing.utils.performancemetrics import PerformanceMetrics


class TestPerformanceMetric(unittest.TestCase):
    def test_basic_metrics(self):
        ground_truth = {1, 2, 3, 4, 5}
        prediction = {2, 6, 4}
        metrics = PerformanceMetrics(truth=ground_truth, prediction=prediction)

        assert metrics.true_positives == {2, 4}
        assert metrics.false_negatives == {1, 3, 5}
        assert metrics.false_positives == {6}

        assert metrics.precision == 2/3
        assert metrics.recall == 2/5
        assert metrics.f1 == 0.5
