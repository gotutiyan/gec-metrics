from .base import MetricBase
import math

class TestMetricBase:
    def test_expected_wins(self):
        metric = MetricBase()
        # Two sentences and three systems
        pairwise_scores = [
            [
                [0, 1, 1],
                [-1, 0, 1],
                [-1, -1, 0]
            ],
            [
                [0, 1, -1],
                [-1, 0, 1],
                [1, -1, 0]
            ]
        ]
        # wins(0, 1) = 2
        # wins(0, 2) = 1
        # wins(1, 0) = 0
        # wins(1, 2) = 2
        # wins(2, 0) = 1
        # wins(2, 1) = 0
        # score(0) = (2/2 + 1/2) / 2 = 0.75
        # score(1) = (0/2 + 2/2) / 2 = 0.50
        # score(2) = (1/2 + 0/2) / 2 = 0.25
        scores = metric.run_expected_wins(pairwise_scores)
        assert all(
            math.isclose(s1, s2, abs_tol=1e-6) 
            for s1, s2 in zip(scores, [0.75, 0.50, 0.25])
        )