from .recommender import MovieRecommender
from .models import (
    UserBasedCF,
    ItemBasedCF,
    SVDModel,
    ContentBasedFilter,
    HybridRecommender
)
from .data_loader import MovieLensDataLoader
from .preprocessing import DataPreprocessor
from .metrics import RecommenderEvaluator

__version__ = '1.0.0'
__author__ = 'techn4r'

__all__ = [
    'MovieRecommender',
    'UserBasedCF',
    'ItemBasedCF',
    'SVDModel',
    'ContentBasedFilter',
    'HybridRecommender',
    'MovieLensDataLoader',
    'DataPreprocessor',
    'RecommenderEvaluator'
]
