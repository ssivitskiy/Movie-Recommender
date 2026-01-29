import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommender import MovieRecommender
from src.metrics import RecommenderEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Movie Recommender')
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='svd',
        choices=['svd', 'user_cf', 'item_cf', 'content', 'hybrid', 'all'],
        help='Model to evaluate'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='ml-100k',
        help='MovieLens dataset'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=4.0,
        help='Rating threshold for relevance'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to saved model'
    )
    
    return parser.parse_args()


def evaluate_model(model_name: str, dataset: str, threshold: float) -> dict:
    """Оценка одной модели"""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name.upper()}")
    print('=' * 60)
    
    recommender = MovieRecommender(dataset=dataset, model=model_name)
    recommender.fit(verbose=False)
    results = recommender.evaluate(threshold=threshold)
    
    return results


def compare_models(dataset: str, threshold: float):
    """Сравнение всех моделей"""
    models = ['svd', 'user_cf', 'item_cf', 'content', 'hybrid']
    all_results = {}
    
    for model_name in models:
        try:
            results = evaluate_model(model_name, dataset, threshold)
            all_results[model_name] = results
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Сводная таблица
    print("\n" + "=" * 80)
    print(" MODEL COMPARISON")
    print("=" * 80)
    
    # Заголовок
    print(f"{'Model':<12} | {'RMSE':>8} | {'MAE':>8} | {'P@10':>8} | {'R@10':>8} | {'NDCG@10':>8} | {'HR@10':>8}")
    print("-" * 80)
    
    # Результаты
    for model_name, results in all_results.items():
        print(f"{model_name:<12} | "
              f"{results.get('rmse', 0):>8.4f} | "
              f"{results.get('mae', 0):>8.4f} | "
              f"{results.get('precision@10', 0):>8.4f} | "
              f"{results.get('recall@10', 0):>8.4f} | "
              f"{results.get('ndcg@10', 0):>8.4f} | "
              f"{results.get('hit_rate@10', 0):>8.4f}")
    
    print("=" * 80)
    
    # Лучшая модель по каждой метрике
    print("\n🏆 Best models by metric:")
    metrics = ['rmse', 'mae', 'precision@10', 'recall@10', 'ndcg@10']
    
    for metric in metrics:
        values = {m: r.get(metric, float('inf') if 'rmse' in metric or 'mae' in metric else 0) 
                  for m, r in all_results.items()}
        
        if 'rmse' in metric or 'mae' in metric:
            best_model = min(values, key=values.get)
        else:
            best_model = max(values, key=values.get)
        
        print(f"  {metric:15} → {best_model} ({values[best_model]:.4f})")
    
    return all_results


def main():
    args = parse_args()
    
    print("=" * 60)
    print(" Movie Recommender Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Relevance threshold: {args.threshold}")
    
    if args.model_path:
        # Загрузка и оценка сохраненной модели
        recommender = MovieRecommender.load(args.model_path)
        recommender.load_data()
        recommender.prepare_data()
        recommender.evaluate(threshold=args.threshold)
        
    elif args.model == 'all':
        # Сравнение всех моделей
        compare_models(args.dataset, args.threshold)
        
    else:
        # Оценка одной модели
        evaluate_model(args.model, args.dataset, args.threshold)


if __name__ == '__main__':
    main()
