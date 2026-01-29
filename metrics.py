import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    recommended_k = recommended[:k]
    relevant_in_k = len(set(recommended_k) & relevant)
    return relevant_in_k / k

def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    recommended_k = recommended[:k]
    relevant_in_k = len(set(recommended_k) & relevant)
    return relevant_in_k / len(relevant)

def f1_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int, relevance_scores: Dict[int, float] = None) -> float:
    if relevance_scores is None:
        relevance_scores = {item: 1.0 for item in relevant}
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevance_scores:
            dcg += relevance_scores[item] / np.log2(i + 2)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
    return dcg / idcg if idcg > 0 else 0.0

def mean_reciprocal_rank(recommended_lists: List[List[int]], relevant_sets: List[Set[int]]) -> float:
    reciprocal_ranks = []
    for recommended, relevant in zip(recommended_lists, relevant_sets):
        rr = 0.0
        for i, item in enumerate(recommended):
            if item in relevant:
                rr = 1.0 / (i + 1)
                break
        reciprocal_ranks.append(rr)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def hit_rate_at_k(recommended_lists: List[List[int]], relevant_sets: List[Set[int]], k: int) -> float:
    hits = 0
    for recommended, relevant in zip(recommended_lists, relevant_sets):
        if len(set(recommended[:k]) & relevant) > 0:
            hits += 1
    return hits / len(recommended_lists) if recommended_lists else 0.0

def coverage(recommended_lists: List[List[int]], n_items: int, k: int) -> float:
    all_recommended = set()
    for recommended in recommended_lists:
        all_recommended.update(recommended[:k])
    return len(all_recommended) / n_items if n_items > 0 else 0.0

def diversity(recommended: List[int], similarity_matrix: np.ndarray) -> float:
    if len(recommended) < 2:
        return 1.0
    total_similarity = 0
    n_pairs = 0
    for i in range(len(recommended)):
        for j in range(i + 1, len(recommended)):
            total_similarity += similarity_matrix[recommended[i], recommended[j]]
            n_pairs += 1
    avg_similarity = total_similarity / n_pairs if n_pairs > 0 else 0
    return 1 - avg_similarity

def novelty(recommended: List[int], item_popularity: Dict[int, float], n_users: int) -> float:
    if len(recommended) == 0:
        return 0.0
    novelty_scores = []
    for item in recommended:
        pop = item_popularity.get(item, 1) / n_users
        novelty_scores.append(-np.log2(pop + 1e-10))
    return np.mean(novelty_scores)

class RecommenderEvaluator:
    """Класс для комплексной оценки рекомендательной системы"""
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        self.k_values = k_values
        
    def evaluate_rating_prediction(self, model, test_data: List[Tuple[int, int, float]]) -> Dict[str, float]:
        y_true, y_pred = [], []
        for user_idx, item_idx, rating in test_data:
            y_true.append(rating)
            y_pred.append(model.predict(user_idx, item_idx))
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return {'rmse': rmse(y_true, y_pred), 'mae': mae(y_true, y_pred)}
    
    def evaluate_ranking(self, model, test_data: Dict[int, Set[int]], n_items: int, threshold: float = 4.0, 
                         item_similarity: np.ndarray = None, item_popularity: Dict[int, float] = None, 
                         n_users: int = None) -> Dict[str, float]:
        results = defaultdict(list)
        recommended_lists, relevant_sets = [], []
        for user_idx, relevant in test_data.items():
            if not relevant: continue
            max_k = max(self.k_values)
            recommendations = model.recommend(user_idx, n=max_k, exclude_known=True)
            recommended = [item for item, _ in recommendations]
            recommended_lists.append(recommended)
            relevant_sets.append(relevant)
            for k in self.k_values:
                results[f'precision@{k}'].append(precision_at_k(recommended, relevant, k))
                results[f'recall@{k}'].append(recall_at_k(recommended, relevant, k))
                results[f'f1@{k}'].append(f1_at_k(recommended, relevant, k))
                results[f'ndcg@{k}'].append(ndcg_at_k(recommended, relevant, k))
            if item_similarity is not None and recommended:
                results['diversity'].append(diversity(recommended[:10], item_similarity))
            if item_popularity is not None and n_users is not None:
                results['novelty'].append(novelty(recommended[:10], item_popularity, n_users))
        final_results = {metric: np.mean(values) for metric, values in results.items()}
        final_results['mrr'] = mean_reciprocal_rank(recommended_lists, relevant_sets)
        for k in self.k_values:
            final_results[f'hit_rate@{k}'] = hit_rate_at_k(recommended_lists, relevant_sets, k)
            final_results[f'coverage@{k}'] = coverage(recommended_lists, n_items, k)
        return final_results
    
    def full_evaluation(self, model, test_ratings: List[Tuple[int, int, float]], n_items: int, threshold: float = 4.0) -> Dict[str, float]:
        rating_metrics = self.evaluate_rating_prediction(model, test_ratings)
        test_relevant = defaultdict(set)
        for user_idx, item_idx, rating in test_ratings:
            if rating >= threshold:
                test_relevant[user_idx].add(item_idx)
        ranking_metrics = self.evaluate_ranking(model, dict(test_relevant), n_items)
        return {**rating_metrics, **ranking_metrics}
    
    def print_results(self, results: Dict[str, float], title: str = "Evaluation Results"):
        print(f"\n{'=' * 50}\n {title}\n{'=' * 50}")
        for metric in ['rmse', 'mae']:
            if metric in results: print(f"  {metric.upper():15} {results[metric]:.4f}")
        for k in self.k_values:
            print(f"\n  K = {k}:")
            for m in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate', 'coverage']:
                print(f"    {m.capitalize():12} {results.get(f'{m}@{k}', 0):.4f}")
        print('=' * 50)
