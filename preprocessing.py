import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

class DataPreprocessor:
    """Предобработка данных для рекомендательной системы"""
    def __init__(self):
        self.user_encoder, self.item_encoder = None, None
        self.user_decoder, self.item_decoder = None, None
        self.n_users, self.n_items, self.global_mean = 0, 0, 0
        
    def fit(self, ratings: pd.DataFrame) -> 'DataPreprocessor':
        u_users, u_items = ratings['user_id'].unique(), ratings['movie_id'].unique()
        self.user_encoder = {u: i for i, u in enumerate(u_users)}
        self.item_encoder = {m: i for i, m in enumerate(u_items)}
        self.user_decoder = {i: u for u, i in self.user_encoder.items()}
        self.item_decoder = {i: m for m, i in self.item_encoder.items()}
        self.n_users, self.n_items = len(u_users), len(u_items)
        self.global_mean = ratings['rating'].mean()
        return self
    
    def transform(self, ratings: pd.DataFrame) -> pd.DataFrame:
        df = ratings.copy()
        df['user_idx'] = df['user_id'].map(self.user_encoder)
        df['item_idx'] = df['movie_id'].map(self.item_encoder)
        df = df.dropna(subset=['user_idx', 'item_idx'])
        df['user_idx'], df['item_idx'] = df['user_idx'].astype(int), df['item_idx'].astype(int)
        return df
    
    def fit_transform(self, ratings: pd.DataFrame) -> pd.DataFrame:
        return self.fit(ratings).transform(ratings)
    
    def to_sparse_matrix(self, ratings: pd.DataFrame) -> csr_matrix:
        if 'user_idx' not in ratings.columns: ratings = self.transform(ratings)
        return csr_matrix((ratings['rating'].values, (ratings['user_idx'].values, ratings['item_idx'].values)),
                          shape=(self.n_users, self.n_items))
    
    def to_dense_matrix(self, ratings: pd.DataFrame, fill_value: float = 0) -> np.ndarray:
        dense = self.to_sparse_matrix(ratings).toarray()
        if fill_value != 0: dense[dense == 0] = fill_value
        return dense

def train_test_split_ratings(ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, 
                             stratify_by_user: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if stratify_by_user:
        train_l, test_l = [], []
        for _, g in ratings.groupby('user_id'):
            if len(g) < 2: train_l.append(g); continue
            nt = max(1, int(len(g) * test_size))
            ti = g.sample(n=nt, random_state=random_state).index
            train_l.append(g.drop(ti)); test_l.append(g.loc[ti])
        return pd.concat(train_l, ignore_index=True), pd.concat(test_l, ignore_index=True)
    return train_test_split(ratings, test_size=test_size, random_state=random_state)

def create_genre_matrix(movies: pd.DataFrame) -> Tuple[np.ndarray, List[str], dict]:
    all_g = set()
    for g in movies['genres'].dropna(): all_g.update(g.split('|'))
    g_names = sorted(list(all_g - {'(no genres listed)', ''}))
    g_to_idx = {g: i for i, g in enumerate(g_names)}
    m_to_idx = {m: i for i, m in enumerate(movies['movie_id'])}
    matrix = np.zeros((len(movies), len(g_names)))
    for _, r in movies.iterrows():
        mi = m_to_idx[r['movie_id']]
        if pd.notna(r['genres']):
            for g in r['genres'].split('|'):
                if g in g_to_idx: matrix[mi, g_to_idx[g]] = 1
    return matrix, g_names, m_to_idx

def filter_cold_start(ratings: pd.DataFrame, min_u: int = 5, min_i: int = 5) -> pd.DataFrame:
    df = ratings.copy()
    while True:
        ilen = len(df)
        uc = df['user_id'].value_counts(); df = df[df['user_id'].isin(uc[uc >= min_u].index)]
        ic = df['movie_id'].value_counts(); df = df[df['movie_id'].isin(ic[ic >= min_i].index)]
        if len(df) == ilen: break
    return df.reset_index(drop=True)
