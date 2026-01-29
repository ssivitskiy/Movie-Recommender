import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class BaseRecommender(ABC):
    """Базовый класс для рекомендательных моделей"""
    
    @abstractmethod
    def fit(self, ratings_matrix: np.ndarray, **kwargs):
        """Обучение модели"""
        pass
    
    @abstractmethod
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Предсказание рейтинга для пары (user, item)"""
        pass
    
    def predict_all(self, user_idx: int) -> np.ndarray:
        """Предсказание рейтингов для всех фильмов"""
        pass
    
    def recommend(self, user_idx: int, n: int = 10, exclude_known: bool = True) -> List[Tuple[int, float]]:
        """Генерация рекомендаций для пользователя"""
        pass


class UserBasedCF(BaseRecommender):
    """User-Based Collaborative Filtering"""
    
    def __init__(self, k_neighbors: int = 50, min_common_items: int = 3):
        """
        Args:
            k_neighbors: Количество соседей для рекомендаций
            min_common_items: Минимальное количество общих оценок
        """
        self.k = k_neighbors
        self.min_common = min_common_items
        self.ratings_matrix = None
        self.user_similarity = None
        self.user_means = None
        self.global_mean = None
        
    def fit(self, ratings_matrix: np.ndarray, **kwargs):
        """
        Обучение модели
        
        Args:
            ratings_matrix: Матрица user-item (n_users x n_items)
        """
        self.ratings_matrix = ratings_matrix.copy()
        
        # Глобальное среднее
        mask = ratings_matrix > 0
        self.global_mean = ratings_matrix[mask].mean() if mask.any() else 0
        
        # Средние рейтинги пользователей
        self.user_means = np.zeros(ratings_matrix.shape[0])
        for i in range(ratings_matrix.shape[0]):
            user_ratings = ratings_matrix[i, ratings_matrix[i] > 0]
            self.user_means[i] = user_ratings.mean() if len(user_ratings) > 0 else self.global_mean
        
        # Центрирование матрицы
        centered_matrix = ratings_matrix.copy()
        for i in range(centered_matrix.shape[0]):
            mask = centered_matrix[i] > 0
            centered_matrix[i, mask] -= self.user_means[i]
        
        # Косинусное сходство между пользователями
        self.user_similarity = cosine_similarity(centered_matrix)
        np.fill_diagonal(self.user_similarity, 0)
        
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Предсказание рейтинга"""
        if self.ratings_matrix is None:
            raise ValueError("Model not fitted")
        
        # Получение соседей, оценивших данный фильм
        item_ratings = self.ratings_matrix[:, item_idx]
        rated_mask = item_ratings > 0
        
        if not rated_mask.any():
            return self.user_means[user_idx]
        
        # Сходство с пользователями, оценившими фильм
        similarities = self.user_similarity[user_idx] * rated_mask
        
        # Топ-k соседей
        top_k_indices = np.argsort(similarities)[-self.k:]
        top_k_indices = top_k_indices[similarities[top_k_indices] > 0]
        
        if len(top_k_indices) == 0:
            return self.user_means[user_idx]
        
        # Взвешенное предсказание
        numerator = 0
        denominator = 0
        
        for neighbor_idx in top_k_indices:
            sim = self.user_similarity[user_idx, neighbor_idx]
            rating_diff = item_ratings[neighbor_idx] - self.user_means[neighbor_idx]
            numerator += sim * rating_diff
            denominator += abs(sim)
        
        if denominator == 0:
            return self.user_means[user_idx]
        
        prediction = self.user_means[user_idx] + numerator / denominator
        return np.clip(prediction, 1, 5)
    
    def predict_all(self, user_idx: int) -> np.ndarray:
        """Предсказание рейтингов для всех фильмов"""
        n_items = self.ratings_matrix.shape[1]
        predictions = np.array([self.predict(user_idx, i) for i in range(n_items)])
        return predictions
    
    def recommend(self, user_idx: int, n: int = 10, exclude_known: bool = True) -> List[Tuple[int, float]]:
        """Генерация рекомендаций"""
        predictions = self.predict_all(user_idx)
        
        if exclude_known:
            known_items = np.where(self.ratings_matrix[user_idx] > 0)[0]
            predictions[known_items] = -np.inf
        
        top_indices = np.argsort(predictions)[-n:][::-1]
        return [(idx, predictions[idx]) for idx in top_indices if predictions[idx] > -np.inf]


class ItemBasedCF(BaseRecommender):
    """Item-Based Collaborative Filtering"""
    
    def __init__(self, k_neighbors: int = 50, min_common_users: int = 3):
        """
        Args:
            k_neighbors: Количество похожих фильмов для рекомендаций
            min_common_users: Минимальное количество общих оценок
        """
        self.k = k_neighbors
        self.min_common = min_common_users
        self.ratings_matrix = None
        self.item_similarity = None
        self.global_mean = None
        
    def fit(self, ratings_matrix: np.ndarray, **kwargs):
        """Обучение модели"""
        self.ratings_matrix = ratings_matrix.copy()
        
        # Глобальное среднее
        mask = ratings_matrix > 0
        self.global_mean = ratings_matrix[mask].mean() if mask.any() else 0
        
        # Косинусное сходство между фильмами (транспонируем матрицу)
        self.item_similarity = cosine_similarity(ratings_matrix.T)
        np.fill_diagonal(self.item_similarity, 0)
        
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Предсказание рейтинга"""
        if self.ratings_matrix is None:
            raise ValueError("Model not fitted")
        
        # Фильмы, оцененные пользователем
        user_ratings = self.ratings_matrix[user_idx]
        rated_mask = user_ratings > 0
        
        if not rated_mask.any():
            return self.global_mean
        
        # Сходство с оцененными фильмами
        similarities = self.item_similarity[item_idx] * rated_mask
        
        # Топ-k похожих фильмов
        top_k_indices = np.argsort(similarities)[-self.k:]
        top_k_indices = top_k_indices[similarities[top_k_indices] > 0]
        
        if len(top_k_indices) == 0:
            return self.global_mean
        
        # Взвешенное предсказание
        numerator = np.sum(similarities[top_k_indices] * user_ratings[top_k_indices])
        denominator = np.sum(np.abs(similarities[top_k_indices]))
        
        if denominator == 0:
            return self.global_mean
        
        prediction = numerator / denominator
        return np.clip(prediction, 1, 5)
    
    def predict_all(self, user_idx: int) -> np.ndarray:
        """Предсказание рейтингов для всех фильмов"""
        n_items = self.ratings_matrix.shape[1]
        predictions = np.array([self.predict(user_idx, i) for i in range(n_items)])
        return predictions
    
    def recommend(self, user_idx: int, n: int = 10, exclude_known: bool = True) -> List[Tuple[int, float]]:
        """Генерация рекомендаций"""
        predictions = self.predict_all(user_idx)
        
        if exclude_known:
            known_items = np.where(self.ratings_matrix[user_idx] > 0)[0]
            predictions[known_items] = -np.inf
        
        top_indices = np.argsort(predictions)[-n:][::-1]
        return [(idx, predictions[idx]) for idx in top_indices if predictions[idx] > -np.inf]
    
    def find_similar_items(self, item_idx: int, n: int = 10) -> List[Tuple[int, float]]:
        """Поиск похожих фильмов"""
        similarities = self.item_similarity[item_idx]
        top_indices = np.argsort(similarities)[-n-1:-1][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]


class SVDModel(BaseRecommender):
    """Matrix Factorization с использованием SVD"""
    
    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        use_bias: bool = True
    ):
        """
        Args:
            n_factors: Количество латентных факторов
            n_epochs: Количество эпох обучения
            lr: Learning rate
            reg: Коэффициент регуляризации
            use_bias: Использовать смещения
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.use_bias = use_bias
        
        self.global_mean = None
        self.user_biases = None
        self.item_biases = None
        self.user_factors = None
        self.item_factors = None
        self.ratings_matrix = None
        
    def fit(self, ratings_matrix: np.ndarray, verbose: bool = True, **kwargs):
        """
        Обучение модели методом SGD
        
        Args:
            ratings_matrix: Матрица рейтингов
            verbose: Вывод прогресса
        """
        self.ratings_matrix = ratings_matrix.copy()
        n_users, n_items = ratings_matrix.shape
        
        # Получение индексов известных рейтингов
        known_ratings = []
        for u in range(n_users):
            for i in range(n_items):
                if ratings_matrix[u, i] > 0:
                    known_ratings.append((u, i, ratings_matrix[u, i]))
        
        # Глобальное среднее
        self.global_mean = np.mean([r for _, _, r in known_ratings])
        
        # Инициализация параметров
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # SGD оптимизация
        for epoch in range(self.n_epochs):
            np.random.shuffle(known_ratings)
            total_loss = 0
            
            for u, i, r in known_ratings:
                # Предсказание
                pred = self.global_mean
                if self.use_bias:
                    pred += self.user_biases[u] + self.item_biases[i]
                pred += np.dot(self.user_factors[u], self.item_factors[i])
                
                # Ошибка
                error = r - pred
                total_loss += error ** 2
                
                # Обновление смещений
                if self.use_bias:
                    self.user_biases[u] += self.lr * (error - self.reg * self.user_biases[u])
                    self.item_biases[i] += self.lr * (error - self.reg * self.item_biases[i])
                
                # Обновление факторов
                user_factor_old = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (error * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (error * user_factor_old - self.reg * self.item_factors[i])
            
            rmse = np.sqrt(total_loss / len(known_ratings))
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")
        
        return self
    
    def fit_svd(self, ratings_matrix: np.ndarray, **kwargs):
        """
        Альтернативное обучение через scipy SVD
        """
        self.ratings_matrix = ratings_matrix.copy()
        n_users, n_items = ratings_matrix.shape
        
        # Заполнение пропусков средним
        mask = ratings_matrix > 0
        self.global_mean = ratings_matrix[mask].mean()
        
        filled_matrix = ratings_matrix.copy()
        filled_matrix[~mask] = self.global_mean
        
        # SVD разложение
        k = min(self.n_factors, min(n_users, n_items) - 1)
        U, sigma, Vt = svds(csr_matrix(filled_matrix), k=k)
        
        # Сохранение факторов
        self.user_factors = U
        self.item_factors = (np.diag(sigma) @ Vt).T
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Предсказание рейтинга"""
        if self.user_factors is None:
            raise ValueError("Model not fitted")
        
        pred = self.global_mean
        if self.use_bias:
            pred += self.user_biases[user_idx] + self.item_biases[item_idx]
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        return np.clip(pred, 1, 5)
    
    def predict_all(self, user_idx: int) -> np.ndarray:
        """Предсказание рейтингов для всех фильмов"""
        predictions = self.global_mean + np.dot(self.user_factors[user_idx], self.item_factors.T)
        if self.use_bias:
            predictions += self.user_biases[user_idx] + self.item_biases
        return np.clip(predictions, 1, 5)
    
    def recommend(self, user_idx: int, n: int = 10, exclude_known: bool = True) -> List[Tuple[int, float]]:
        """Генерация рекомендаций"""
        predictions = self.predict_all(user_idx)
        
        if exclude_known:
            known_items = np.where(self.ratings_matrix[user_idx] > 0)[0]
            predictions[known_items] = -np.inf
        
        top_indices = np.argsort(predictions)[-n:][::-1]
        return [(idx, predictions[idx]) for idx in top_indices if predictions[idx] > -np.inf]


class ContentBasedFilter(BaseRecommender):
    """Content-Based Filtering по жанрам"""
    
    def __init__(self, k_similar: int = 20):
        """
        Args:
            k_similar: Количество похожих фильмов для рекомендаций
        """
        self.k = k_similar
        self.genre_matrix = None
        self.content_similarity = None
        self.ratings_matrix = None
        
    def fit(self, ratings_matrix: np.ndarray, genre_matrix: np.ndarray = None, **kwargs):
        """
        Обучение модели
        
        Args:
            ratings_matrix: Матрица рейтингов
            genre_matrix: Матрица жанров (n_items x n_genres)
        """
        self.ratings_matrix = ratings_matrix.copy()
        
        if genre_matrix is not None:
            self.genre_matrix = genre_matrix
            self.content_similarity = cosine_similarity(genre_matrix)
            np.fill_diagonal(self.content_similarity, 0)
        
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Предсказание рейтинга"""
        if self.content_similarity is None:
            raise ValueError("Model not fitted or genre_matrix not provided")
        
        user_ratings = self.ratings_matrix[user_idx]
        rated_mask = user_ratings > 0
        
        if not rated_mask.any():
            return 3.0  # Нейтральный рейтинг
        
        # Сходство с оцененными фильмами
        similarities = self.content_similarity[item_idx] * rated_mask
        
        # Топ-k похожих фильмов
        top_k_indices = np.argsort(similarities)[-self.k:]
        top_k_indices = top_k_indices[similarities[top_k_indices] > 0]
        
        if len(top_k_indices) == 0:
            return user_ratings[rated_mask].mean()
        
        # Взвешенное предсказание
        numerator = np.sum(similarities[top_k_indices] * user_ratings[top_k_indices])
        denominator = np.sum(similarities[top_k_indices])
        
        if denominator == 0:
            return user_ratings[rated_mask].mean()
        
        return np.clip(numerator / denominator, 1, 5)
    
    def predict_all(self, user_idx: int) -> np.ndarray:
        """Предсказание рейтингов для всех фильмов"""
        n_items = self.ratings_matrix.shape[1]
        return np.array([self.predict(user_idx, i) for i in range(n_items)])
    
    def recommend(self, user_idx: int, n: int = 10, exclude_known: bool = True) -> List[Tuple[int, float]]:
        """Генерация рекомендаций"""
        predictions = self.predict_all(user_idx)
        
        if exclude_known:
            known_items = np.where(self.ratings_matrix[user_idx] > 0)[0]
            predictions[known_items] = -np.inf
        
        top_indices = np.argsort(predictions)[-n:][::-1]
        return [(idx, predictions[idx]) for idx in top_indices if predictions[idx] > -np.inf]
    
    def build_user_profile(self, user_idx: int) -> np.ndarray:
        """Построение профиля пользователя на основе жанров"""
        if self.genre_matrix is None:
            raise ValueError("Genre matrix not provided")
        
        user_ratings = self.ratings_matrix[user_idx]
        rated_mask = user_ratings > 0
        
        if not rated_mask.any():
            return np.zeros(self.genre_matrix.shape[1])
        
        # Взвешенный профиль жанров
        weighted_genres = self.genre_matrix[rated_mask].T @ user_ratings[rated_mask]
        profile = weighted_genres / np.sum(user_ratings[rated_mask])
        
        return profile


class HybridRecommender(BaseRecommender):
    """Гибридная рекомендательная система"""
    
    def __init__(
        self,
        models: Dict[str, BaseRecommender] = None,
        weights: Dict[str, float] = None
    ):
        """
        Args:
            models: Словарь моделей {'name': model}
            weights: Веса моделей {'name': weight}
        """
        self.models = models or {}
        self.weights = weights or {}
        self.ratings_matrix = None
        
    def add_model(self, name: str, model: BaseRecommender, weight: float = 1.0):
        """Добавление модели"""
        self.models[name] = model
        self.weights[name] = weight
        
    def fit(self, ratings_matrix: np.ndarray, **kwargs):
        """Обучение всех моделей"""
        self.ratings_matrix = ratings_matrix
        
        for name, model in self.models.items():
            print(f"Fitting {name}...")
            model.fit(ratings_matrix, **kwargs)
        
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Взвешенное предсказание"""
        total_weight = sum(self.weights.values())
        prediction = 0
        
        for name, model in self.models.items():
            pred = model.predict(user_idx, item_idx)
            prediction += (self.weights[name] / total_weight) * pred
        
        return prediction
    
    def predict_all(self, user_idx: int) -> np.ndarray:
        """Предсказание для всех фильмов"""
        total_weight = sum(self.weights.values())
        n_items = self.ratings_matrix.shape[1]
        predictions = np.zeros(n_items)
        
        for name, model in self.models.items():
            preds = model.predict_all(user_idx)
            predictions += (self.weights[name] / total_weight) * preds
        
        return predictions
    
    def recommend(self, user_idx: int, n: int = 10, exclude_known: bool = True) -> List[Tuple[int, float]]:
        """Генерация рекомендаций"""
        predictions = self.predict_all(user_idx)
        
        if exclude_known:
            known_items = np.where(self.ratings_matrix[user_idx] > 0)[0]
            predictions[known_items] = -np.inf
        
        top_indices = np.argsort(predictions)[-n:][::-1]
        return [(idx, predictions[idx]) for idx in top_indices if predictions[idx] > -np.inf]


if __name__ == '__main__':
    # Тестирование моделей
    np.random.seed(42)
    
    # Генерация тестовых данных
    n_users, n_items = 100, 50
    ratings = np.random.randint(0, 6, (n_users, n_items)).astype(float)
    ratings[ratings == 0] = 0  # Пропуски
    
    # User-Based CF
    print("Testing User-Based CF...")
    user_cf = UserBasedCF(k_neighbors=20)
    user_cf.fit(ratings)
    recs = user_cf.recommend(0, n=5)
    print(f"Recommendations: {recs[:3]}")
    
    # Item-Based CF
    print("\nTesting Item-Based CF...")
    item_cf = ItemBasedCF(k_neighbors=20)
    item_cf.fit(ratings)
    recs = item_cf.recommend(0, n=5)
    print(f"Recommendations: {recs[:3]}")
    
    # SVD
    print("\nTesting SVD...")
    svd = SVDModel(n_factors=20, n_epochs=10)
    svd.fit(ratings, verbose=True)
    recs = svd.recommend(0, n=5)
    print(f"Recommendations: {recs[:3]}")
