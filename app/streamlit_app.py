import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommender import MovieRecommender

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .movie-card {
        padding: 1rem;
        border-radius: 10px;
        background: #f8f9fa;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender(model_type: str = 'svd'):
    """Загрузка и кэширование модели"""
    recommender = MovieRecommender(model=model_type)
    recommender.fit(verbose=False)
    return recommender

def display_movie_card(title: str, genres: str, rating: float, rank: int = None):
    rank_str = f"**#{rank}** " if rank else ""
    stars = "⭐" * int(round(rating))
    genres_list = genres.split('|') if pd.notna(genres) else []
    genres_html = ' '.join([f'`{g}`' for g in genres_list[:4]])
    
    st.markdown(f"""
    <div class="movie-card">
        <h4>{rank_str}{title}</h4>
        <p>Predicted Rating: {rating:.2f} {stars}</p>
        <p>{genres_html}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎬 Movie Recommender System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        model_type = st.selectbox(
            "Model Type",
            options=['svd', 'user_cf', 'item_cf', 'hybrid'],
            format_func=lambda x: {
                'svd': '📊 SVD (Matrix Factorization)',
                'user_cf': '👥 User-Based CF',
                'item_cf': '🎥 Item-Based CF',
                'hybrid': '🔀 Hybrid'
            }.get(x, x)
        )
        n_recommendations = st.slider("Number of recommendations", 5, 30, 10)
        st.markdown("---")
        st.markdown("### 📚 About")
        st.info("Personalized movie recommendations using MovieLens dataset and Collaborative Filtering.")

    with st.spinner(f"Loading {model_type} model..."):
        recommender = load_recommender(model_type)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendations", "🔍 Similar Movies", "📊 Popular", "📜 History"])
    
    with tab1:
        st.header("🎯 Personalized Recommendations")
        col1, col2 = st.columns([1, 3])
        with col1:
            user_id = st.number_input("Enter User ID", 1, 943, 1)
            get_recs = st.button("🎬 Get Recommendations", type="primary", use_container_width=True)
        with col2:
            if get_recs:
                try:
                    recs = recommender.recommend_for_user(user_id=user_id, n=n_recommendations)
                    col_a, col_b = st.columns(2)
                    for i, (_, row) in enumerate(recs.iterrows()):
                        with col_a if i % 2 == 0 else col_b:
                            display_movie_card(row['title'], row['genres'], row['predicted_rating'], rank=i+1)
                except ValueError as e:
                    st.error(f"Error: {e}")
    
    with tab2:
        st.header("🔍 Find Similar Movies")
        all_movies = recommender.movies['title'].tolist()
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_movie = st.selectbox("Select a movie", options=all_movies[:500])
            manual_input = st.text_input("Or enter title manually:")
            find_similar = st.button("🔍 Find Similar", type="primary", use_container_width=True)
        with col2:
            if find_similar:
                movie_to_search = manual_input if manual_input else selected_movie
                try:
                    similar = recommender.find_similar_movies(movie_to_search, n=n_recommendations)
                    for _, row in similar.iterrows():
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"**{row['title']}**")
                            st.caption(f"Genres: {row['genres']}")
                        with c2:
                            st.metric("Similarity", f"{row['similarity']:.3f}")
                        st.divider()
                except ValueError as e:
                    st.error(f"Error: {e}")

    with tab3:
        st.header("📊 Most Popular Movies")
        popular = recommender.get_popular_movies(n=20)
        for i, row in popular.iterrows():
            col1, col2, col3, col4 = st.columns([0.5, 3, 1, 1])
            with col1: st.markdown(f"**{i+1}**")
            with col2:
                st.markdown(f"**{row['title']}**")
                st.caption(row['genres'])
            with col3: st.metric("Rating", f"⭐ {row['avg_rating']:.2f}")
            with col4: st.metric("Reviews", int(row['n_ratings']))
            st.divider()

    with tab4:
        st.header("📜 User Rating History")
        uid = st.number_input("User ID", 1, 943, 1, key="history_user")
        if st.button("📜 Show History"):
            history = recommender.get_user_history(uid)
            if not history.empty:
                st.success(f"User {uid} rated {len(history)} movies")
                st.dataframe(history[['title', 'rating', 'genres']].head(20), use_container_width=True)
            else:
                st.warning("No ratings found")

    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #888;'>
        <p>Built by <a href='https://github.com/techn4r'>techn4r</a> | 2026</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
