import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    rmse, mae, precision_at_k, recall_at_k, f1_at_k,
    ndcg_at_k, mean_reciprocal_rank, hit_rate_at_k,
    coverage, diversity, novelty
)

class TestRatingMetrics:
    def test_rmse_perfect(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert rmse(y_true, y_pred) == 0.0

    def test_rmse_calculation(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        assert rmse(y_true, y_pred) == 1.0

    def test_mae_perfect(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert mae(y_true, y_pred) == 0.0

class TestRankingMetrics:
    def test_precision_at_k_partial(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5, 7, 9}
        assert precision_at_k(recommended, relevant, 5) == 0.6

    def test_recall_at_k_all_found(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        assert recall_at_k(recommended, relevant, 5) == 1.0

    def test_ndcg_at_k_perfect_ranking(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        assert ndcg_at_k(recommended, relevant, 5) == 1.0

class TestEdgeCases:
    def test_empty_recommendations(self):
        assert precision_at_k([], {1, 2, 3}, 0) == 0.0

    def test_k_larger_than_list(self):
        """Исправлено: Precision@10 для 3 элементов = 0.3"""
        recommended = [1, 2, 3]
        relevant = {1, 2, 3, 4, 5}
        p = precision_at_k(recommended, relevant, 10)
        assert p == 0.3
