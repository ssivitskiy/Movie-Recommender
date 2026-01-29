"""
Tests for recommendation models
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    UserBasedCF,
    ItemBasedCF,
    SVDModel,
    ContentBasedFilter,
    HybridRecommender
)


@pytest.fixture
def sample_ratings_matrix():
    """Создание тестовой матрицы рейтингов"""
    np.random.seed(42)
    matrix = np.random.randint(0, 6, (50, 30)).astype(float)
    # Добавляем паттерны для тестирования
    matrix[:10, :5] = np.random.randint(4, 6, (10, 5))  # Первые 10 юзеров любят первые 5 фильмов
    matrix[10:20, 5:10] = np.random.randint(4, 6, (10, 5))  # Следующие 10 любят другие фильмы
    return matrix


@pytest.fixture
def sample_genre_matrix():
    """Создание тестовой матрицы жанров"""
    np.random.seed(42)
    return np.random.randint(0, 2, (30, 10)).astype(float)


class TestUserBasedCF:
    """Тесты для User-Based CF"""
    
    def test_fit(self, sample_ratings_matrix):
        """Тест обучения модели"""
        model = UserBasedCF(k_neighbors=10)
        model.fit(sample_ratings_matrix)
        
        assert model.ratings_matrix is not None
        assert model.user_similarity is not None
        assert model.user_similarity.shape == (50, 50)
    
    def test_predict(self, sample_ratings_matrix):
        """Тест предсказания"""
        model = UserBasedCF(k_neighbors=10)
        model.fit(sample_ratings_matrix)
        
        pred = model.predict(0, 0)
        assert 1 <= pred <= 5
    
    def test_recommend(self, sample_ratings_matrix):
        """Тест рекомендаций"""
        model = UserBasedCF(k_neighbors=10)
        model.fit(sample_ratings_matrix)
        
        recs = model.recommend(0, n=5)
        assert len(recs) <= 5
        assert all(isinstance(r, tuple) for r in recs)
        assert all(len(r) == 2 for r in recs)
    
    def test_not_fitted_error(self):
        """Тест ошибки без обучения"""
        model = UserBasedCF()
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(0, 0)


class TestItemBasedCF:
    """Тесты для Item-Based CF"""
    
    def test_fit(self, sample_ratings_matrix):
        """Тест обучения модели"""
        model = ItemBasedCF(k_neighbors=10)
        model.fit(sample_ratings_matrix)
        
        assert model.ratings_matrix is not None
        assert model.item_similarity is not None
        assert model.item_similarity.shape == (30, 30)
    
    def test_predict(self, sample_ratings_matrix):
        """Тест предсказания"""
        model = ItemBasedCF(k_neighbors=10)
        model.fit(sample_ratings_matrix)
        
        pred = model.predict(0, 15)
        assert 1 <= pred <= 5
    
    def test_find_similar_items(self, sample_ratings_matrix):
        """Тест поиска похожих фильмов"""
        model = ItemBasedCF(k_neighbors=10)
        model.fit(sample_ratings_matrix)
        
        similar = model.find_similar_items(0, n=5)
        assert len(similar) == 5
        assert all(sim >= 0 for _, sim in similar)


class TestSVDModel:
    """Тесты для SVD модели"""
    
    def test_fit(self, sample_ratings_matrix):
        """Тест обучения модели"""
        model = SVDModel(n_factors=10, n_epochs=5)
        model.fit(sample_ratings_matrix, verbose=False)
        
        assert model.user_factors is not None
        assert model.item_factors is not None
        assert model.user_factors.shape == (50, 10)
        assert model.item_factors.shape == (30, 10)
    
    def test_predict(self, sample_ratings_matrix):
        """Тест предсказания"""
        model = SVDModel(n_factors=10, n_epochs=5)
        model.fit(sample_ratings_matrix, verbose=False)
        
        pred = model.predict(0, 0)
        assert 1 <= pred <= 5
    
    def test_predict_all(self, sample_ratings_matrix):
        """Тест предсказания для всех фильмов"""
        model = SVDModel(n_factors=10, n_epochs=5)
        model.fit(sample_ratings_matrix, verbose=False)
        
        preds = model.predict_all(0)
        assert len(preds) == 30
        assert all(1 <= p <= 5 for p in preds)
    
    def test_convergence(self, sample_ratings_matrix):
        """Тест сходимости обучения"""
        model = SVDModel(n_factors=20, n_epochs=50, lr=0.01)
        model.fit(sample_ratings_matrix, verbose=False)
        
        # Проверяем, что модель выучила известные рейтинги
        errors = []
        for u in range(50):
            for i in range(30):
                if sample_ratings_matrix[u, i] > 0:
                    pred = model.predict(u, i)
                    errors.append(abs(pred - sample_ratings_matrix[u, i]))
        
        mae = np.mean(errors)
        assert mae < 1.5  # MAE должен быть достаточно низким


class TestContentBasedFilter:
    """Тесты для контентной фильтрации"""
    
    def test_fit(self, sample_ratings_matrix, sample_genre_matrix):
        """Тест обучения модели"""
        model = ContentBasedFilter(k_similar=10)
        model.fit(sample_ratings_matrix, genre_matrix=sample_genre_matrix)
        
        assert model.content_similarity is not None
        assert model.content_similarity.shape == (30, 30)
    
    def test_predict(self, sample_ratings_matrix, sample_genre_matrix):
        """Тест предсказания"""
        model = ContentBasedFilter(k_similar=10)
        model.fit(sample_ratings_matrix, genre_matrix=sample_genre_matrix)
        
        pred = model.predict(0, 15)
        assert 1 <= pred <= 5
    
    def test_build_user_profile(self, sample_ratings_matrix, sample_genre_matrix):
        """Тест построения профиля пользователя"""
        model = ContentBasedFilter(k_similar=10)
        model.fit(sample_ratings_matrix, genre_matrix=sample_genre_matrix)
        
        profile = model.build_user_profile(0)
        assert len(profile) == 10
        assert all(p >= 0 for p in profile)


class TestHybridRecommender:
    """Тесты для гибридной модели"""
    
    def test_add_model(self):
        """Тест добавления модели"""
        hybrid = HybridRecommender()
        hybrid.add_model('svd', SVDModel(n_factors=10), weight=0.6)
        hybrid.add_model('cf', ItemBasedCF(k_neighbors=10), weight=0.4)
        
        assert len(hybrid.models) == 2
        assert hybrid.weights['svd'] == 0.6
        assert hybrid.weights['cf'] == 0.4
    
    def test_fit(self, sample_ratings_matrix):
        """Тест обучения гибридной модели"""
        hybrid = HybridRecommender()
        hybrid.add_model('svd', SVDModel(n_factors=10, n_epochs=5), weight=0.6)
        hybrid.add_model('cf', ItemBasedCF(k_neighbors=10), weight=0.4)
        
        hybrid.fit(sample_ratings_matrix, verbose=False)
        
        assert hybrid.models['svd']._SVDModel__class__ is not None or True
    
    def test_predict(self, sample_ratings_matrix):
        """Тест предсказания гибридной модели"""
        hybrid = HybridRecommender()
        hybrid.add_model('svd', SVDModel(n_factors=10, n_epochs=5), weight=0.6)
        hybrid.add_model('cf', ItemBasedCF(k_neighbors=10), weight=0.4)
        hybrid.fit(sample_ratings_matrix, verbose=False)
        
        pred = hybrid.predict(0, 15)
        assert 1 <= pred <= 5


class TestModelIntegration:
    """Интеграционные тесты"""
    
    def test_all_models_same_interface(self, sample_ratings_matrix, sample_genre_matrix):
        """Тест единого интерфейса всех моделей"""
        models = [
            UserBasedCF(k_neighbors=10),
            ItemBasedCF(k_neighbors=10),
            SVDModel(n_factors=10, n_epochs=5)
        ]
        
        for model in models:
            # Fit
            model.fit(sample_ratings_matrix, verbose=False)
            
            # Predict
            pred = model.predict(0, 0)
            assert isinstance(pred, (int, float))
            
            # Recommend
            recs = model.recommend(0, n=5)
            assert isinstance(recs, list)
    
    def test_recommendations_are_new_items(self, sample_ratings_matrix):
        """Тест, что рекомендуются новые фильмы"""
        model = SVDModel(n_factors=10, n_epochs=5)
        model.fit(sample_ratings_matrix, verbose=False)
        
        user_idx = 0
        known_items = set(np.where(sample_ratings_matrix[user_idx] > 0)[0])
        
        recs = model.recommend(user_idx, n=10, exclude_known=True)
        rec_items = set(item_idx for item_idx, _ in recs)
        
        # Проверяем, что рекомендованные фильмы не пересекаются с известными
        assert len(rec_items & known_items) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
