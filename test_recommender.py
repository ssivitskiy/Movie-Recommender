"""
Tests for MovieRecommender class
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMovieRecommenderUnit:
    """Юнит-тесты для MovieRecommender"""
    
    def test_init_default_model(self):
        """Тест инициализации с моделью по умолчанию"""
        from src.recommender import MovieRecommender
        recommender = MovieRecommender(model='svd')
        assert recommender.model_name == 'svd'
        assert recommender._fitted == False
    
    def test_init_different_models(self):
        """Тест инициализации разных моделей"""
        from src.recommender import MovieRecommender
        
        models = ['svd', 'user_cf', 'item_cf', 'content', 'hybrid']
        for model_name in models:
            recommender = MovieRecommender(model=model_name)
            assert recommender.model is not None
    
    def test_init_unknown_model_raises(self):
        """Тест ошибки при неизвестной модели"""
        from src.recommender import MovieRecommender
        
        with pytest.raises(ValueError, match="Unknown model"):
            MovieRecommender(model='unknown_model')
    
    def test_recommend_without_fit_raises(self):
        """Тест ошибки при рекомендациях без обучения"""
        from src.recommender import MovieRecommender
        
        recommender = MovieRecommender(model='svd')
        with pytest.raises(ValueError, match="Model not fitted"):
            recommender.recommend_for_user(user_id=1)
    
    def test_predict_without_fit_raises(self):
        """Тест ошибки при предсказании без обучения"""
        from src.recommender import MovieRecommender
        
        recommender = MovieRecommender(model='svd')
        with pytest.raises(ValueError, match="Model not fitted"):
            recommender.predict_rating(user_id=1, movie_title="Test")


class TestDataPreprocessor:
    """Тесты для препроцессора данных"""
    
    @pytest.fixture
    def sample_ratings(self):
        """Создание тестовых рейтингов"""
        return pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'movie_id': [10, 20, 10, 30, 20],
            'rating': [5.0, 4.0, 3.0, 5.0, 4.0]
        })
    
    def test_fit(self, sample_ratings):
        """Тест обучения препроцессора"""
        from src.preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_ratings)
        
        assert preprocessor.n_users == 3
        assert preprocessor.n_items == 3
        assert len(preprocessor.user_encoder) == 3
        assert len(preprocessor.item_encoder) == 3
    
    def test_transform(self, sample_ratings):
        """Тест преобразования данных"""
        from src.preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        transformed = preprocessor.fit_transform(sample_ratings)
        
        assert 'user_idx' in transformed.columns
        assert 'item_idx' in transformed.columns
        assert transformed['user_idx'].max() < preprocessor.n_users
        assert transformed['item_idx'].max() < preprocessor.n_items
    
    def test_to_sparse_matrix(self, sample_ratings):
        """Тест создания разреженной матрицы"""
        from src.preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_ratings)
        sparse = preprocessor.to_sparse_matrix(sample_ratings)
        
        assert sparse.shape == (3, 3)
        assert sparse.nnz == 5  # 5 ненулевых элементов


class TestTrainTestSplit:
    """Тесты для разделения данных"""
    
    @pytest.fixture
    def sample_ratings(self):
        """Создание тестовых рейтингов"""
        np.random.seed(42)
        return pd.DataFrame({
            'user_id': np.repeat(range(1, 11), 10),
            'movie_id': np.tile(range(1, 11), 10),
            'rating': np.random.randint(1, 6, 100).astype(float)
        })
    
    def test_split_ratio(self, sample_ratings):
        """Тест соотношения train/test"""
        from src.preprocessing import train_test_split_ratings
        
        train, test = train_test_split_ratings(sample_ratings, test_size=0.2)
        
        total = len(train) + len(test)
        assert abs(len(test) / total - 0.2) < 0.1  # ~20% в тесте
    
    def test_no_data_leak(self, sample_ratings):
        """Тест отсутствия утечки данных"""
        from src.preprocessing import train_test_split_ratings
        
        train, test = train_test_split_ratings(sample_ratings, test_size=0.2)
        
        # Проверяем, что нет одинаковых записей
        train_keys = set(zip(train['user_id'], train['movie_id']))
        test_keys = set(zip(test['user_id'], test['movie_id']))
        
        assert len(train_keys & test_keys) == 0


class TestFilterColdStart:
    """Тесты для фильтрации холодного старта"""
    
    def test_filter_removes_sparse_users(self):
        """Тест удаления пользователей с малым числом оценок"""
        from src.preprocessing import filter_cold_start
        
        ratings = pd.DataFrame({
            'user_id': [1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3],
            'movie_id': [1, 2, 3, 4, 5, 1, 1, 2, 3, 4, 5],
            'rating': [5.0] * 11
        })
        
        filtered = filter_cold_start(ratings, min_user_ratings=3, min_item_ratings=1)
        
        # User 2 имеет только 1 оценку, должен быть удален
        assert 2 not in filtered['user_id'].values
        assert 1 in filtered['user_id'].values
        assert 3 in filtered['user_id'].values


class TestGenreMatrix:
    """Тесты для матрицы жанров"""
    
    def test_create_genre_matrix(self):
        """Тест создания матрицы жанров"""
        from src.preprocessing import create_genre_matrix
        
        movies = pd.DataFrame({
            'movie_id': [1, 2, 3],
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'genres': ['Action|Comedy', 'Drama', 'Action|Drama']
        })
        
        matrix, genre_names, movie_to_idx = create_genre_matrix(movies)
        
        assert matrix.shape[0] == 3  # 3 фильма
        assert len(genre_names) == 3  # Action, Comedy, Drama
        assert matrix.sum() == 5  # Всего 5 жанровых меток


class TestDataLoader:
    """Тесты для загрузчика данных"""
    
    def test_datasets_config(self):
        """Тест конфигурации датасетов"""
        from src.data_loader import MovieLensDataLoader
        
        loader = MovieLensDataLoader('ml-100k')
        assert 'ml-100k' in loader.DATASETS
        assert 'url' in loader.config
    
    def test_invalid_dataset_raises(self):
        """Тест ошибки при неверном датасете"""
        from src.data_loader import MovieLensDataLoader
        
        with pytest.raises(ValueError, match="Unknown dataset"):
            MovieLensDataLoader('invalid-dataset')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
