# %% [markdown]
# # 🎬 Movie Recommender: Exploratory Data Analysis
# Exploratory analysis of the MovieLens dataset for techn4r/movie-recommender project.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import MovieLensDataLoader

# Modern plotting configuration
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = [12, 6]

# %%
loader = MovieLensDataLoader('ml-100k')
ratings, movies = loader.load_data()
stats = loader.get_stats()

for key, value in stats.items():
    print(f"{key:25} {value}")

# %% [markdown]
# ## Rating Distribution
# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
sns.countplot(data=ratings, x='rating', ax=axes[0], color='steelblue')
axes[0].set_title('Distribution of Ratings')

# Time analysis
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings.set_index('datetime').resample('M')['rating'].mean().plot(ax=axes[1], marker='o')
axes[1].set_title('Average Rating Over Time')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## User & Movie Activity
# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

user_counts = ratings.groupby('user_id').size()
sns.histplot(user_counts, bins=50, ax=axes[0], color='coral', kde=True)
axes[0].set_title('Ratings per User')

movie_counts = ratings.groupby('movie_id').size()
sns.histplot(movie_counts, bins=50, ax=axes[1], color='mediumpurple', kde=True)
axes[1].set_title('Ratings per Movie')
plt.show()

# %% [markdown]
# ## Genre Analysis
# %%
all_genres = []
for genres in movies['genres'].dropna():
    all_genres.extend(genres.split('|'))

genre_counts = pd.Series(all_genres).value_counts()
sns.barplot(x=genre_counts.values, y=genre_counts.index, color='teal')
plt.title('Distribution of Movie Genres')
plt.show()

# %% [markdown]
# ## Matrix Sparsity Visualization
# %%
sample_users = ratings['user_id'].unique()[:100]
sample_movies = ratings['movie_id'].unique()[:100]
sample_df = ratings[ratings['user_id'].isin(sample_users) & ratings['movie_id'].isin(sample_movies)]
matrix = sample_df.pivot(index='user_id', columns='movie_id', values='rating').notnull()

plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap='YlGnBu', cbar=False)
plt.title(f'Sparsity Matrix (Sample 100x100). Global Sparsity: {stats["sparsity"]:.2%}')
plt.show()

# %%
print("EDA Complete! ✅ All data prepared for techn4r/movie-recommender.")
