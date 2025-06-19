import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.metrics import silhouette_score
from gensim.models import CoherenceModel
from gensim import corpora
import scipy.spatial.distance as distance


class TopicEvaluator:
    """класс для оценки качества тематических моделей"""
    
    def __init__(self):
        self.metrics = {}
    
    def compute_coherence(self, 
                         topics: List[List[str]], 
                         texts: List[List[str]],
                         measure: str = 'c_v') -> float:
        """вычисление кохерентности тем"""
        if not topics or not texts:
            return 0.0
        
        try:
            # создаем словарь для gensim
            dictionary = corpora.Dictionary(texts)
            
            # фильтруем темы - оставляем только слова, которые есть в словаре
            filtered_topics = []
            for topic in topics:
                filtered_words = [word for word in topic if word in dictionary.token2id]
                if len(filtered_words) >= 2:  # минимум 2 слова для coherence
                    filtered_topics.append(filtered_words)
            
            if not filtered_topics:
                print(f"Нет валидных тем для расчета coherence ({measure})")
                return 0.0
            
            # создаем модель кохерентности
            coherence_model = CoherenceModel(
                topics=filtered_topics,
                texts=texts,
                dictionary=dictionary,
                coherence=measure
            )
            
            return coherence_model.get_coherence()
            
        except Exception as e:
            print(f"Ошибка при расчете coherence ({measure}): {e}")
            return 0.0
    
    def compute_silhouette(self, 
                          embeddings: np.ndarray, 
                          labels: List[int],
                          sample_size: int = 10000) -> float:
        """вычисление силуэта кластеризации"""
        if len(set(labels)) < 2:  # нужно минимум 2 кластера
            return 0.0
        
        # фильтруем outliers (-1)
        valid_indices = [i for i, label in enumerate(labels) if label != -1]
        
        if len(valid_indices) < 2:
            return 0.0
        
        valid_embeddings = embeddings[valid_indices]
        valid_labels = [labels[i] for i in valid_indices]
        
        # для больших данных используем выборку
        if len(valid_embeddings) > sample_size:
            indices = np.random.choice(len(valid_embeddings), sample_size, replace=False)
            valid_embeddings = valid_embeddings[indices]
            valid_labels = [valid_labels[i] for i in indices]
        
        try:
            return silhouette_score(valid_embeddings, valid_labels)
        except:
            return 0.0
    
    def compute_topic_diversity(self, 
                               topics: List[List[Tuple[str, float]]], 
                               top_k: int = 10) -> float:
        """вычисление разнообразия тем (доля уникальных слов)"""
        if not topics:
            return 0.0
        
        all_words = []
        unique_words = set()
        
        for topic in topics:
            words = [word for word, _ in topic[:top_k]]
            all_words.extend(words)
            unique_words.update(words)
        
        if len(all_words) == 0:
            return 0.0
        
        return len(unique_words) / len(all_words)
    
    def compute_topic_coverage(self, 
                              doc_topics: List[int], 
                              outlier_label: int = -1) -> Dict[str, float]:
        """вычисление покрытия тем"""
        total_docs = len(doc_topics)
        covered_docs = sum(1 for topic in doc_topics if topic != outlier_label)
        
        coverage = covered_docs / total_docs if total_docs > 0 else 0.0
        outlier_ratio = 1.0 - coverage
        
        return {
            'coverage': coverage,
            'outlier_ratio': outlier_ratio,
            'covered_docs': covered_docs,
            'total_docs': total_docs
        }
    
    def compute_topic_sizes(self, doc_topics: List[int]) -> Dict[str, Any]:
        """анализ размеров тем"""
        topic_counts = {}
        for topic in doc_topics:
            if topic != -1:  # исключаем outliers
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        if not topic_counts:
            return {
                'num_topics': 0,
                'avg_size': 0,
                'min_size': 0,
                'max_size': 0,
                'std_size': 0,
                'topic_distribution': {}
            }
        
        sizes = list(topic_counts.values())
        
        return {
            'num_topics': len(topic_counts),
            'avg_size': np.mean(sizes),
            'min_size': np.min(sizes),
            'max_size': np.max(sizes),
            'std_size': np.std(sizes),
            'topic_distribution': topic_counts
        }
    
    def evaluate_model(self, 
                      topics: List[List[Tuple[str, float]]],
                      texts: List[List[str]],
                      embeddings: np.ndarray,
                      doc_topics: List[int],
                      model_name: str = "model") -> Dict[str, Any]:
        """комплексная оценка модели"""
        results = {'model': model_name}
        
        # преобразуем темы для coherence - берем только слова, убираем веса
        topic_words = []
        for topic in topics:
            if topic:  # проверяем что тема не пустая
                words = []
                for word, _ in topic:
                    if isinstance(word, str):
                        # Разбиваем фразы на отдельные токены (для BERTopic)
                        if ' ' in word:
                            # Это фраза - разбиваем на токены
                            tokens = word.split()
                            words.extend(tokens)
                        else:
                            # Это отдельный токен
                            words.append(word)
                
                # Убираем дубликаты и пустые строки
                words = list(dict.fromkeys([w for w in words if w.strip()]))
                
                if words:  # добавляем только непустые списки слов
                    topic_words.append(words)
        
        # кохерентность - только если есть темы и тексты
        if topic_words and texts:
            try:
                results['coherence_cv'] = self.compute_coherence(topic_words, texts, 'c_v')
            except Exception as e:
                print(f"Ошибка при расчете coherence_cv для {model_name}: {e}")
                results['coherence_cv'] = 0.0
            
            try:
                results['coherence_umass'] = self.compute_coherence(topic_words, texts, 'u_mass')
            except Exception as e:
                print(f"Ошибка при расчете coherence_umass для {model_name}: {e}")
                results['coherence_umass'] = 0.0
        else:
            print(f"Нет тем или текстов для расчета coherence для {model_name}")
            results['coherence_cv'] = 0.0
            results['coherence_umass'] = 0.0
        
        # силуэт
        results['silhouette'] = self.compute_silhouette(embeddings, doc_topics)
        
        # разнообразие тем
        results['topic_diversity'] = self.compute_topic_diversity(topics)
        
        # покрытие
        coverage_stats = self.compute_topic_coverage(doc_topics)
        results.update(coverage_stats)
        
        # размеры тем
        size_stats = self.compute_topic_sizes(doc_topics)
        results.update(size_stats)
        
        return results


def compare_models(lda_results: Dict[str, Any], 
                   bertopic_results: Dict[str, Any]) -> pd.DataFrame:
    """сравнение результатов двух моделей"""
    
    comparison_metrics = [
        'coherence_cv', 'coherence_umass', 'silhouette',
        'topic_diversity', 'coverage', 'outlier_ratio',
        'num_topics', 'avg_size'
    ]
    
    comparison_data = []
    
    for metric in comparison_metrics:
        lda_value = lda_results.get(metric, 0)
        bertopic_value = bertopic_results.get(metric, 0)
        
        comparison_data.append({
            'metric': metric,
            'LDA': lda_value,
            'BERTopic': bertopic_value,
            'difference': bertopic_value - lda_value if isinstance(bertopic_value, (int, float)) and isinstance(lda_value, (int, float)) else None
        })
    
    return pd.DataFrame(comparison_data)


def compute_topic_similarity(topics1: List[List[str]], 
                            topics2: List[List[str]],
                            method: str = 'jaccard') -> np.ndarray:
    """вычисление матрицы сходства между темами разных моделей"""
    
    def jaccard_similarity(set1: set, set2: set) -> float:
        """коэффициент Жаккара"""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def overlap_coefficient(set1: set, set2: set) -> float:
        """коэффициент перекрытия"""
        intersection = len(set1.intersection(set2))
        min_size = min(len(set1), len(set2))
        return intersection / min_size if min_size > 0 else 0.0
    
    similarity_matrix = np.zeros((len(topics1), len(topics2)))
    
    for i, topic1 in enumerate(topics1):
        set1 = set(topic1)
        for j, topic2 in enumerate(topics2):
            set2 = set(topic2)
            
            if method == 'jaccard':
                similarity_matrix[i, j] = jaccard_similarity(set1, set2)
            elif method == 'overlap':
                similarity_matrix[i, j] = overlap_coefficient(set1, set2)
    
    return similarity_matrix


def find_topic_matches(similarity_matrix: np.ndarray, 
                      threshold: float = 0.3) -> List[Tuple[int, int, float]]:
    """поиск соответствующих тем между моделями"""
    matches = []
    
    # находим максимальные сходства для каждой темы
    for i in range(similarity_matrix.shape[0]):
        max_j = np.argmax(similarity_matrix[i, :])
        max_similarity = similarity_matrix[i, max_j]
        
        if max_similarity >= threshold:
            matches.append((i, max_j, max_similarity))
    
    return matches 