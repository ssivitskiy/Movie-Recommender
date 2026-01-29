"""
Tests for evaluation metrics
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    rmse,
    mae,
    precision_at_k,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
    coverage,
    diversity,
    novelty
)


class TestRatingMetrics:
    """Тесты для метрик предсказания рейтингов"""
    
    def test_rmse_perfect(self):
        """Тест RMSE при идеальном предсказании"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert rmse(y_true, y_pred) == 0.0
    
    def test_rmse_calculation(self):
        """Тест вычисления RMSE"""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        # MSE = (1 + 1 + 1) / 3 = 1, RMSE = 1
        assert rmse(y_true, y_pred) == 1.0
    
    def test_mae_perfect(self):
        """Тест MAE при идеальном предсказании"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert mae(y_true, y_pred) == 0.0
    
    def test_mae_calculation(self):
        """Тест вычисления MAE"""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 5])
        # MAE = (1 + 1 + 2) / 3 = 4/3
        assert abs(mae(y_true, y_pred) - 4/3) < 1e-10


class TestRankingMetrics:
    """Тесты для метрик ранжирования"""
    
    def test_precision_at_k_all_relevant(self):
        """Тест Precision@K когда все релевантны"""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}
        assert precision_at_k(recommended, relevant, 5) == 1.0
    
    def test_precision_at_k_none_relevant(self):
        """Тест Precision@K когда ничего не релевантно"""
        recommended = [1, 2, 3, 4, 5]
        relevant = {6, 7, 8}
        assert precision_at_k(recommended, relevant, 5) == 0.0
    
    def test_precision_at_k_partial(self):
        """Тест Precision@K с частичной релевантностью"""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5, 7, 9}
        # 3 из 5 релевантны
        assert precision_at_k(recommended, relevant, 5) == 0.6
    
    def test_recall_at_k_all_found(self):
        """Тест Recall@K когда все найдены"""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        assert recall_at_k(recommended, relevant, 5) == 1.0
    
    def test_recall_at_k_partial(self):
        """Тест Recall@K с частичным покрытием"""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 6, 8, 10}
        # 2 из 5 релевантных найдены
        assert recall_at_k(recommended, relevant, 5) == 0.4
    
    def test_recall_at_k_empty_relevant(self):
        """Тест Recall@K с пустым множеством релевантных"""
        recommended = [1, 2, 3]
        relevant = set()
        assert recall_at_k(recommended, relevant, 3) == 0.0
    
    def test_f1_at_k_calculation(self):
        """Тест вычисления F1@K"""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 6, 8, 10}
        
        p = precision_at_k(recommended, relevant, 5)  # 0.4
        r = recall_at_k(recommended, relevant, 5)  # 0.4
        expected_f1 = 2 * p * r / (p + r)  # 0.4
        
        assert abs(f1_at_k(recommended, relevant, 5) - expected_f1) < 1e-10
    
    def test_f1_at_k_zero(self):
        """Тест F1@K когда precision и recall равны нулю"""
        recommended = [1, 2, 3]
        relevant = {4, 5, 6}
        assert f1_at_k(recommended, relevant, 3) == 0.0
    
    def test_ndcg_at_k_perfect_ranking(self):
        """Тест NDCG@K при идеальном ранжировании"""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        # Первые 3 элемента релевантны - идеальное ранжирование
        assert ndcg_at_k(recommended, relevant, 5) == 1.0
    
    def test_ndcg_at_k_worst_ranking(self):
        """Тест NDCG@K при плохом ранжировании"""
        recommended = [4, 5, 6, 1, 2]
        relevant = {1, 2, 3}
        # Релевантные элементы в конце списка
        ndcg = ndcg_at_k(recommended, relevant, 5)
        assert 0 < ndcg < 1.0
    
    def test_ndcg_at_k_no_relevant(self):
        """Тест NDCG@K без релевантных элементов"""
        recommended = [1, 2, 3]
        relevant = set()
        assert ndcg_at_k(recommended, relevant, 3) == 0.0


class TestAggregatedMetrics:
    """Тесты для агрегированных метрик"""
    
    def test_mrr_first_hit(self):
        """Тест MRR когда первый элемент релевантен"""
        recommended_lists = [[1, 2, 3], [4, 5, 6]]
        relevant_sets = [{1, 7}, {4, 8}]
        # RR = 1 для обоих пользователей
        assert mean_reciprocal_rank(recommended_lists, relevant_sets) == 1.0
    
    def test_mrr_second_hit(self):
        """Тест MRR когда второй элемент релевантен"""
        recommended_lists = [[1, 2, 3]]
        relevant_sets = [{2}]
        # RR = 1/2
        assert mean_reciprocal_rank(recommended_lists, relevant_sets) == 0.5
    
    def test_mrr_no_hit(self):
        """Тест MRR без попаданий"""
        recommended_lists = [[1, 2, 3]]
        relevant_sets = [{4, 5, 6}]
        assert mean_reciprocal_rank(recommended_lists, relevant_sets) == 0.0
    
    def test_hit_rate_all_hit(self):
        """Тест Hit Rate когда все попали"""
        recommended_lists = [[1, 2], [3, 4]]
        relevant_sets = [{1}, {4}]
        assert hit_rate_at_k(recommended_lists, relevant_sets, 2) == 1.0
    
    def test_hit_rate_partial(self):
        """Тест Hit Rate с частичными попаданиями"""
        recommended_lists = [[1, 2], [3, 4]]
        relevant_sets = [{1}, {5}]
        assert hit_rate_at_k(recommended_lists, relevant_sets, 2) == 0.5
    
    def test_coverage_full(self):
        """Тест Coverage когда все фильмы рекомендованы"""
        recommended_lists = [[1, 2], [3, 4], [5, 6]]
        assert coverage(recommended_lists, n_items=6, k=2) == 1.0
    
    def test_coverage_partial(self):
        """Тест Coverage с частичным покрытием"""
        recommended_lists = [[1, 2], [1, 3]]
        # Уникальные: 1, 2, 3 - 3 из 10
        assert coverage(recommended_lists, n_items=10, k=2) == 0.3


class TestDiversityMetrics:
    """Тесты для метрик разнообразия"""
    
    def test_diversity_identical_items(self):
        """Тест Diversity для идентичных элементов"""
        # Создаем матрицу с полным сходством
        similarity = np.ones((5, 5))
        np.fill_diagonal(similarity, 0)
        
        recommended = [0, 1, 2]
        div = diversity(recommended, similarity)
        assert div == 0.0  # Нет разнообразия
    
    def test_diversity_different_items(self):
        """Тест Diversity для разных элементов"""
        similarity = np.zeros((5, 5))
        np.fill_diagonal(similarity, 1)
        
        recommended = [0, 1, 2]
        div = diversity(recommended, similarity)
        assert div == 1.0  # Максимальное разнообразие
    
    def test_novelty_calculation(self):
        """Тест вычисления Novelty"""
        item_popularity = {0: 100, 1: 10, 2: 1}
        n_users = 1000
        
        # Рекомендуем популярный элемент
        popular_novelty = novelty([0], item_popularity, n_users)
        
        # Рекомендуем непопулярный элемент
        unpopular_novelty = novelty([2], item_popularity, n_users)
        
        # Непопулярный должен иметь большую новизну
        assert unpopular_novelty > popular_novelty


class TestEdgeCases:
    """Тесты граничных случаев"""
    
    def test_empty_recommendations(self):
        """Тест с пустыми рекомендациями"""
        assert precision_at_k([], {1, 2, 3}, 0) == 0.0
        assert recall_at_k([], {1, 2, 3}, 0) == 0.0
    
    def test_k_larger_than_list(self):
        """Тест когда K больше размера списка"""
        recommended = [1, 2, 3]
        relevant = {1, 2, 3, 4, 5}
        
        p = precision_at_k(recommended, relevant, 10)
        assert p == 1.0  # 3/3 в топ-3
    
    def test_single_item(self):
        """Тест с одним элементом"""
        recommended = [1]
        relevant = {1}
        
        assert precision_at_k(recommended, relevant, 1) == 1.0
        assert recall_at_k(recommended, relevant, 1) == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
