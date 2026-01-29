import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

class MovieLensDataLoader:
    """Загрузчик данных MovieLens"""
    
    DATASETS = {
        'ml-100k': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
            'ratings_file': 'u.data',
            'movies_file': 'u.item',
            'ratings_sep': '\t',
            'movies_sep': '|',
            'movies_encoding': 'latin-1'
        },
        'ml-1m': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
            'ratings_file': 'ratings.dat',
            'movies_file': 'movies.dat',
            'ratings_sep': '::',
            'movies_sep': '::',
            'movies_encoding': 'latin-1'
        },
        'ml-latest-small': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            'ratings_file': 'ratings.csv',
            'movies_file': 'movies.csv',
            'ratings_sep': ',',
            'movies_sep': ',',
            'movies_encoding': 'utf-8'
        }
    }
    
    GENRE_COLUMNS_100K = [
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    def __init__(self, dataset: str = 'ml-100k', data_dir: str = 'data'):
        """
        Инициализация загрузчика
        
        Args:
            dataset: Название датасета ('ml-100k', 'ml-1m', 'ml-latest-small')
            data_dir: Директория для хранения данных
        """
        if dataset not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(self.DATASETS.keys())}")
        
        self.dataset = dataset
        self.config = self.DATASETS[dataset]
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, force: bool = False) -> Path:
        """
        Скачивание датасета
        """
        dataset_dir = self.raw_dir / self.dataset
        zip_path = self.raw_dir / f"{self.dataset}.zip"
        
        if dataset_dir.exists() and not force:
            return dataset_dir
        
        urllib.request.urlretrieve(self.config['url'], zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        
        zip_path.unlink()
        return dataset_dir
    
    def load_ratings(self) -> pd.DataFrame:
        """
        Загрузка рейтингов
        """
        dataset_dir = self.raw_dir / self.dataset
        ratings_path = dataset_dir / self.config['ratings_file']
        
        if not ratings_path.exists():
            self.download()
        
        if self.dataset == 'ml-100k':
            df = pd.read_csv(
                ratings_path,
                sep=self.config['ratings_sep'],
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                engine='python'
            )
        elif self.dataset == 'ml-1m':
            df = pd.read_csv(
                ratings_path,
                sep=self.config['ratings_sep'],
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                engine='python'
            )
        else:
            df = pd.read_csv(ratings_path)
            df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        
        return df
    
    def load_movies(self) -> pd.DataFrame:
        """
        Загрузка информации о фильмах
        """
        dataset_dir = self.raw_dir / self.dataset
        movies_path = dataset_dir / self.config['movies_file']
        
        if not movies_path.exists():
            self.download()
        
        if self.dataset == 'ml-100k':
            columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + self.GENRE_COLUMNS_100K
            df = pd.read_csv(
                movies_path,
                sep=self.config['movies_sep'],
                names=columns,
                encoding=self.config['movies_encoding'],
                engine='python'
            )
            df['genres'] = df[self.GENRE_COLUMNS_100K].apply(
                lambda row: '|'.join([g for g, v in zip(self.GENRE_COLUMNS_100K, row) if v == 1]),
                axis=1
            )
            df = df[['movie_id', 'title', 'genres']]
            
        elif self.dataset == 'ml-1m':
            df = pd.read_csv(
                movies_path,
                sep=self.config['movies_sep'],
                names=['movie_id', 'title', 'genres'],
                encoding=self.config['movies_encoding'],
                engine='python'
            )
        else:
            df = pd.read_csv(movies_path)
            df.columns = ['movie_id', 'title', 'genres']
        
        return df
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.load_ratings(), self.load_movies()
    
    def get_stats(self) -> dict:
        ratings, movies = self.load_data()
        return {
            'n_users': ratings['user_id'].nunique(),
            'n_movies': ratings['movie_id'].nunique(),
            'n_ratings': len(ratings),
            'rating_range': (ratings['rating'].min(), ratings['rating'].max()),
            'avg_rating': ratings['rating'].mean(),
            'sparsity': 1 - len(ratings) / (ratings['user_id'].nunique() * ratings['movie_id'].nunique()),
            'avg_ratings_per_user': len(ratings) / ratings['user_id'].nunique(),
            'avg_ratings_per_movie': len(ratings) / ratings['movie_id'].nunique()
        }

def create_user_item_matrix(ratings: pd.DataFrame, fillna: float = 0) -> Tuple[pd.DataFrame, dict, dict]:
    matrix = ratings.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating',
        fill_value=fillna
    )
    user_to_idx = {user: idx for idx, user in enumerate(matrix.index)}
    item_to_idx = {item: idx for idx, item in enumerate(matrix.columns)}
    return matrix, user_to_idx, item_to_idx

if __name__ == '__main__':
    loader = MovieLensDataLoader('ml-100k')
    loader.download()
    stats = loader.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
