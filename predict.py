import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.recommender import MovieRecommender

def parse_args():
    parser = argparse.ArgumentParser(description='Get Movie Recommendations')
    parser.add_argument('--user-id', '-u', type=int, required=True)
    parser.add_argument('--top-n', '-n', type=int, default=10)
    parser.add_argument('--model-path', '-m', type=str, default=None)
    parser.add_argument('--similar-to', '-s', type=str, default=None)
    parser.add_argument('--show-history', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    recommender = MovieRecommender()
    
    if args.model_path:
        recommender.load_model(args.model_path)
    else:
        recommender.fit()
        
    if args.show_history:
        history = recommender.get_user_history(args.user_id)
        for _, row in history.head(10).iterrows():
            print(f"  {row['title'][:45]:45} | ⭐ {row['rating']:.1f}")

    print(f"\n🎬 Top {args.top_n} Recommendations for User {args.user_id}:")
    try:
        recs = recommender.recommend_for_user(user_id=args.user_id, n=args.top_n)
        for i, row in recs.iterrows():
            genres = row['genres'][:25] if pd.notna(row['genres']) else 'N/A'
            print(f"  {i+1:2}. {row['title'][:40]:40} | {genres:25} | ⭐ {row['predicted_rating']:.2f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
