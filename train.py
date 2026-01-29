import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.recommender import MovieRecommender

def parse_args():
    p = argparse.ArgumentParser(description='Train Movie Recommender')
    p.add_argument('--model', '-m', type=str, default='svd', choices=['svd', 'user_cf', 'item_cf', 'content', 'hybrid'])
    p.add_argument('--dataset', '-d', type=str, default='ml-100k', choices=['ml-100k', 'ml-1m', 'ml-latest-small'])
    p.add_argument('--epochs', '-e', type=int, default=20)
    p.add_argument('--factors', '-f', type=int, default=100)
    p.add_argument('--neighbors', '-k', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--reg', type=float, default=0.02)
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--min-user-ratings', type=int, default=5)
    p.add_argument('--min-item-ratings', type=int, default=5)
    p.add_argument('--save', '-s', type=str, default=None)
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--verbose', '-v', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    params = {}
    if args.model == 'svd': params = {'n_factors': args.factors, 'n_epochs': args.epochs, 'lr': args.lr, 'reg': args.reg}
    elif args.model in ['user_cf', 'item_cf']: params = {'k_neighbors': args.neighbors}
    elif args.model == 'content': params = {'k_similar': args.neighbors}
    
    recommender = MovieRecommender(dataset=args.dataset, model=args.model, model_params=params)
    recommender.load_data(min_user_ratings=args.min_user_ratings, min_item_ratings=args.min_item_ratings)
    recommender.prepare_data(test_size=args.test_size)
    recommender.fit(verbose=args.verbose)
    if args.evaluate: recommender.evaluate()
    if args.save: recommender.save(args.save)
    else: recommender.save()
    recs = recommender.recommend_for_user(user_id=1, n=5)
    for _, r in recs.iterrows(): print(f"  {r['title'][:50]:50} | Rating: {r['predicted_rating']:.2f}")

if __name__ == '__main__': main()
