#!/usr/bin/env python3
"""
Основной скрипт для запуска пайплайна тематического моделирования
ИДЕАЛЬНЫЙ ПЛАН: Единая система метрик + дополнительное сравнение с фиксированным количеством тем
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import warnings

# добавляем путь к нашим модулям
sys.path.append('src')

from data_preprocessing import load_and_clean_data, TextPreprocessor
from lda_model import LDAModel, tune_lda_topics
from bertopic_model import BERTopicModel, tune_bertopic_clustering
from evaluation import TopicEvaluator, compare_models, find_optimal_topic_model
from visualization import TopicVisualizer, visualize_embeddings_comparison, visualize_metrics_comparison, visualize_comprehensive_comparison

warnings.filterwarnings("ignore", category=DeprecationWarning) 

# --- Константы ---
DATA_PATH = "data/raw/data_for_analysis.csv"
EMBEDDINGS_CACHE = "models/embeddings_cache.pkl"
RESULTS_PATH = "results/"
FIGURES_PATH = os.path.join(RESULTS_PATH, "figures")
TABLES_PATH = os.path.join(RESULTS_PATH, "tables")

def ensure_dir(path):
    """Убедиться, что директория существует."""
    os.makedirs(path, exist_ok=True)

def load_or_create_embeddings(documents: List[str], 
                             embeddings_path: str = 'models/embeddings.pkl') -> np.ndarray:
    """Загрузка или создание эмбеддингов с кэшированием"""
    if os.path.exists(embeddings_path):
        print("📂 Загружаем сохраненные эмбеддинги...")
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"✅ Загружено {embeddings.shape[0]} эмбеддингов размерности {embeddings.shape[1]}")
        return embeddings
    else:
        print("🔄 Создаем новые эмбеддинги...")
        from sentence_transformers import SentenceTransformer
        
        # используем многоязычную модель для русского языка
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(documents, show_progress_bar=True)
        
        # сохраняем для будущего использования
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"✅ Создано и сохранено {embeddings.shape[0]} эмбеддингов")
        return embeddings


def evaluate_with_composite_metrics(topics: List[List[str]], 
                                   texts: List[List[str]], 
                                   embeddings: np.ndarray,
                                   doc_topics: List[int],
                                   evaluator: TopicEvaluator) -> Dict[str, float]:
    """
    Единая система оценки с композитными метриками
    Основано на академических исследованиях (Egger & Yu 2022, Meaney et al. 2023)
    """
    # базовые метрики
    coherence_cv = evaluator.compute_coherence(topics, texts, 'c_v')
    coherence_umass = evaluator.compute_coherence(topics, texts, 'u_mass') 
    silhouette = evaluator.compute_silhouette(embeddings, doc_topics)
    coverage_info = evaluator.compute_topic_coverage(doc_topics)
    coverage = coverage_info['coverage']
    
    # композитная оценка (основана на Meaney et al. 2023)
    # Веса подобраны на основе важности метрик в академической литературе
    composite_score = (
        0.35 * coherence_cv +      # семантическая связность (основная метрика)
        0.25 * silhouette +        # качество кластеризации  
        0.25 * coverage +          # покрытие документов
        0.15 * (-coherence_umass)  # разнообразие тем (u_mass обычно отрицательная)
    )
    
    return {
        'coherence_cv': coherence_cv,
        'coherence_umass': coherence_umass,
        'silhouette': silhouette,
        'coverage': coverage,
        'composite_score': composite_score
    }


def run_preprocessing(data_path: str, save_path: str = 'data/processed') -> tuple:
    """запуск предобработки данных"""
    print("=== ЭТАП 1: Предобработка данных ===")
    
    # создаем папку для результатов
    os.makedirs(save_path, exist_ok=True)
    
    # загружаем и очищаем данные
    print("загрузка исходных данных...")
    df = load_and_clean_data(data_path)
    print(f"загружено {len(df)} документов")
    
    # предобработка текстов
    print("предобработка текстов...")
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_corpus(df['text'].tolist())
    
    # фильтруем пустые документы
    valid_indices = [i for i, tokens in enumerate(processed_texts) if len(tokens) > 0]
    df_filtered = df.iloc[valid_indices].copy()
    processed_texts_filtered = [processed_texts[i] for i in valid_indices]
    
    print(f"после фильтрации: {len(df_filtered)} документов")
    
    # сохраняем результаты
    df_filtered.to_csv(f'{save_path}/processed_reviews.csv', index=False)
    with open(f'{save_path}/processed_texts.pkl', 'wb') as f:
        pickle.dump(processed_texts_filtered, f)
    
    print("предобработка завершена")
    return df_filtered, processed_texts_filtered


def run_lda_modeling_with_metrics(processed_texts: List[List[str]], 
                                 embeddings: np.ndarray,
                                 save_path: str = 'models') -> Dict[str, Any]:
    """LDA моделирование с единой системой метрик"""
    print("\n=== ЭТАП 2: LDA моделирование (по композитным метрикам) ===")
    
    os.makedirs(save_path, exist_ok=True)
    evaluator = TopicEvaluator()
    
    # диапазон тем для тестирования
    topic_range = list(range(10, 36, 5))  # [10, 15, 20, 25, 30, 35]
    print(f"тестируем количество тем: {topic_range}")
    
    tuning_results = []
    best_score = -float('inf')
    best_model = None
    best_topics = 0
    
    for num_topics in topic_range:
        print(f"\n🔄 Тестируем {num_topics} тем...")
        
        # обучаем модель
        lda_model = LDAModel(random_state=42)
        lda_model.prepare_corpus(processed_texts)
        lda_model.train(num_topics=num_topics)
        
        # получаем результаты
        topics = lda_model.get_topics()
        doc_topics = lda_model.get_dominant_topics()
        
        # преобразуем темы для оценки
        topic_words = [[word for word, _ in topic] for topic in topics]
        
        # оцениваем по композитным метрикам
        metrics = evaluate_with_composite_metrics(
            topic_words, processed_texts, embeddings, doc_topics, evaluator
        )
        
        # сохраняем результат
        result = {
            'num_topics': num_topics,
            **metrics
        }
        tuning_results.append(result)
        
        print(f"  📊 Композитная оценка: {metrics['composite_score']:.3f}")
        print(f"     - Coherence CV: {metrics['coherence_cv']:.3f}")
        print(f"     - Silhouette: {metrics['silhouette']:.3f}")
        print(f"     - Coverage: {metrics['coverage']:.3f}")
        
        # обновляем лучшую модель
        if metrics['composite_score'] > best_score:
            best_score = metrics['composite_score']
            best_model = lda_model
            best_topics = num_topics
    
    print(f"\n🏆 Лучший результат: {best_topics} тем (композитная оценка: {best_score:.3f})")
    
    # сохраняем лучшую модель и результаты
    best_model.save_model(f'{save_path}/lda_model_optimal.pkl')
    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(f'{save_path}/lda_tuning_metrics.csv', index=False)
    
    return {
        'model': best_model,
        'tuning_results': tuning_df,
        'best_topics': best_topics,
        'best_score': best_score
    }


def run_bertopic_modeling_with_metrics(df: pd.DataFrame, 
                                      embeddings: np.ndarray,
                                      save_path: str = 'models') -> Dict[str, Any]:
    """BERTopic моделирование с единой системой метрик"""
    print("\n=== ЭТАП 3: BERTopic моделирование (по композитным метрикам) ===")
    
    documents = df['text'].tolist()
    evaluator = TopicEvaluator()
    
    # параметры для тестирования
    cluster_sizes = [5, 10, 15, 20, 25]
    print(f"тестируем min_cluster_size: {cluster_sizes}")
    
    tuning_results = []
    best_score = -float('inf')
    best_model = None
    best_cluster_size = 0
    
    for cluster_size in cluster_sizes:
        print(f"\n🔄 Тестируем min_cluster_size={cluster_size}...")
        
        # обучаем модель
        bertopic_model = BERTopicModel(random_state=42)
        bertopic_model.train(
            documents, 
            min_cluster_size=cluster_size,
            nr_topics="auto",
            calculate_probabilities=True,
            embeddings=embeddings  # используем готовые эмбеддинги
        )
        
        # получаем результаты
        topics = bertopic_model.get_topics()
        doc_topics = bertopic_model.get_document_topics()
        
        # преобразуем темы для оценки (убираем веса и разбиваем фразы)
        topic_words = []
        for topic in topics:
            words = []
            for word, _ in topic:
                if ' ' in word:  # фраза - разбиваем
                    words.extend(word.split())
                else:
                    words.append(word)
            topic_words.append(list(dict.fromkeys(words)))  # убираем дубликаты
        
        # получаем обработанные тексты для coherence
        preprocessor = TextPreprocessor()
        processed_texts = preprocessor.preprocess_corpus(documents)
        
        # оцениваем по композитным метрикам
        metrics = evaluate_with_composite_metrics(
            topic_words, processed_texts, embeddings, doc_topics, evaluator
        )
        
        # дополнительная информация
        num_topics = len([t for t in set(doc_topics) if t != -1])
        coverage_info = evaluator.compute_topic_coverage(doc_topics)
        
        # сохраняем результат
        result = {
            'min_cluster_size': cluster_size,
            'num_topics': num_topics,
            'outlier_ratio': coverage_info['outlier_ratio'],
            **metrics
        }
        tuning_results.append(result)
        
        print(f"  📊 Композитная оценка: {metrics['composite_score']:.3f}")
        print(f"     - Количество тем: {num_topics}")
        print(f"     - Coherence CV: {metrics['coherence_cv']:.3f}")
        print(f"     - Silhouette: {metrics['silhouette']:.3f}")
        print(f"     - Coverage: {metrics['coverage']:.3f}")
        
        # обновляем лучшую модель
        if metrics['composite_score'] > best_score:
            best_score = metrics['composite_score']
            best_model = bertopic_model
            best_cluster_size = cluster_size
    
    print(f"\n🏆 Лучший результат: min_cluster_size={best_cluster_size} (композитная оценка: {best_score:.3f})")
    
    # сохраняем лучшую модель и результаты
    best_model.save_model(f'{save_path}/bertopic_model_optimal.pkl')
    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(f'{save_path}/bertopic_tuning_metrics.csv', index=False)
    
    return {
        'model': best_model,
        'tuning_results': tuning_df,
        'best_cluster_size': best_cluster_size,
        'best_score': best_score
    }


def run_additional_lda_comparison(processed_texts: List[List[str]], 
                                 bertopic_num_topics: int,
                                 embeddings: np.ndarray,
                                 save_path: str = 'models') -> Dict[str, Any]:
    """
    ДОПОЛНИТЕЛЬНОЕ ИССЛЕДОВАНИЕ: LDA с фиксированным количеством тем как у BERTopic
    Для справедливого сравнения моделей при одинаковом количестве тем
    """
    print(f"\n=== ЭТАП 4: Дополнительное сравнение LDA с {bertopic_num_topics} темами ===")
    print("🔍 Цель: сравнить модели при одинаковом количестве тем")
    
    evaluator = TopicEvaluator()
    
    # обучаем LDA с фиксированным количеством тем
    print(f"🔄 Обучаем LDA с {bertopic_num_topics} темами...")
    lda_fixed = LDAModel(random_state=42)
    lda_fixed.prepare_corpus(processed_texts)
    lda_fixed.train(num_topics=bertopic_num_topics)
    
    # получаем результаты
    topics = lda_fixed.get_topics()
    doc_topics = lda_fixed.get_dominant_topics()
    topic_words = [[word for word, _ in topic] for topic in topics]
    
    # оцениваем по композитным метрикам
    metrics = evaluate_with_composite_metrics(
        topic_words, processed_texts, embeddings, doc_topics, evaluator
    )
    
    print(f"📊 Результаты LDA с {bertopic_num_topics} темами:")
    print(f"   - Композитная оценка: {metrics['composite_score']:.3f}")
    print(f"   - Coherence CV: {metrics['coherence_cv']:.3f}")
    print(f"   - Silhouette: {metrics['silhouette']:.3f}")
    print(f"   - Coverage: {metrics['coverage']:.3f}")
    
    # сохраняем модель
    lda_fixed.save_model(f'{save_path}/lda_model_fixed_{bertopic_num_topics}.pkl')
    
    return {
        'model': lda_fixed,
        'num_topics': bertopic_num_topics,
        'metrics': metrics
    }


def run_comprehensive_evaluation(lda_optimal: Dict[str, Any],
                               bertopic_optimal: Dict[str, Any], 
                               lda_fixed: Dict[str, Any],
                               processed_texts: List[List[str]],
                               df: pd.DataFrame,
                               embeddings: np.ndarray,
                               save_path: str = 'report') -> Dict[str, Any]:
    """Комплексная оценка всех моделей"""
    print("\n=== ЭТАП 5: Комплексная оценка и сравнение ===")
    
    os.makedirs(save_path, exist_ok=True)
    evaluator = TopicEvaluator()
    
    # Подготавливаем данные для сравнения
    models_comparison = []
    
    # 1. LDA оптимальная (по метрикам)
    lda_opt_topics = [[word for word, _ in topic] for topic in lda_optimal['model'].get_topics()]
    lda_opt_doc_topics = lda_optimal['model'].get_dominant_topics()
    lda_opt_metrics = evaluate_with_composite_metrics(
        lda_opt_topics, processed_texts, embeddings, lda_opt_doc_topics, evaluator
    )
    
    models_comparison.append({
        'model': 'LDA_Optimal',
        'num_topics': lda_optimal['best_topics'],
        'selection_method': 'Композитные метрики',
        **lda_opt_metrics
    })
    
    # 2. BERTopic оптимальная (по метрикам)
    bertopic_topics = bertopic_optimal['model'].get_topics()
    bertopic_doc_topics = bertopic_optimal['model'].get_document_topics()
    
    # преобразуем BERTopic темы
    bertopic_topic_words = []
    for topic in bertopic_topics:
        words = []
        for word, _ in topic:
            if ' ' in word:
                words.extend(word.split())
            else:
                words.append(word)
        bertopic_topic_words.append(list(dict.fromkeys(words)))
    
    bertopic_metrics = evaluate_with_composite_metrics(
        bertopic_topic_words, processed_texts, embeddings, bertopic_doc_topics, evaluator
    )
    
    bertopic_num_topics = len([t for t in set(bertopic_doc_topics) if t != -1])
    models_comparison.append({
        'model': 'BERTopic_Optimal',
        'num_topics': bertopic_num_topics,
        'selection_method': 'Композитные метрики',
        **bertopic_metrics
    })
    
    # 3. LDA с фиксированным количеством тем (дополнительное сравнение)
    models_comparison.append({
        'model': f'LDA_Fixed_{lda_fixed["num_topics"]}',
        'num_topics': lda_fixed['num_topics'],
        'selection_method': 'Фиксированное количество (как BERTopic)',
        **lda_fixed['metrics']
    })
    
    # создаем DataFrame для сравнения
    comparison_df = pd.DataFrame(models_comparison)
    
    # добавляем ранжирование по композитной оценке
    comparison_df['rank_composite'] = comparison_df['composite_score'].rank(ascending=False)
    comparison_df['rank_coherence'] = comparison_df['coherence_cv'].rank(ascending=False)
    comparison_df['rank_silhouette'] = comparison_df['silhouette'].rank(ascending=False)
    comparison_df['rank_coverage'] = comparison_df['coverage'].rank(ascending=False)
    
    # сохраняем результаты
    comparison_df.to_csv(f'{save_path}/comprehensive_model_comparison.csv', index=False)
    
    # детальные результаты
    detailed_results = {
        'lda_optimal': lda_optimal,
        'bertopic_optimal': bertopic_optimal,
        'lda_fixed': lda_fixed,
        'comparison': comparison_df,
        'academic_justification': {
            'composite_metrics_source': 'Meaney et al. (2023), Egger & Yu (2022)',
            'weights_rationale': {
                'coherence_cv': 0.35,  # основная метрика семантической связности
                'silhouette': 0.25,    # качество кластеризации
                'coverage': 0.25,      # покрытие документов
                'coherence_umass': 0.15  # разнообразие тем
            },
            'fixed_topics_rationale': 'Дополнительное сравнение для устранения влияния количества тем на результаты'
        }
    }
    
    with open(f'{save_path}/detailed_evaluation_results.pkl', 'wb') as f:
        pickle.dump(detailed_results, f)
    
    print("📊 Результаты сравнения:")
    print(comparison_df[['model', 'num_topics', 'composite_score', 'coherence_cv', 'silhouette', 'coverage']].round(3))
    
    return detailed_results


def create_enhanced_visualizations(results: Dict[str, Any],
                                 df: pd.DataFrame,
                                 processed_texts: List[List[str]],
                                 embeddings: np.ndarray,
                                 save_path: str = 'report/figures') -> None:
    """Создание расширенных визуализаций"""
    print("\n=== ЭТАП 6: Создание визуализаций ===")
    
    os.makedirs(save_path, exist_ok=True)
    viz = TopicVisualizer()
    
    # 1. Графики подбора параметров с композитными метриками
    print("📈 Графики подбора параметров...")
    lda_tuning = results['lda_optimal']['tuning_results']
    bertopic_tuning = results['bertopic_optimal']['tuning_results']
    
    viz.plot_composite_metrics_tuning(lda_tuning, 'LDA')
    viz.plot_composite_metrics_tuning(bertopic_tuning, 'BERTopic')
    
    # 2. Сравнение всех моделей
    print("📊 Сравнение моделей...")
    comparison_df = results['comparison']
    viz.plot_comprehensive_comparison(comparison_df)
    
    # 3. Детальный анализ тем
    print("🔍 Анализ тем...")
    lda_opt_model = results['lda_optimal']['model']
    bertopic_opt_model = results['bertopic_optimal']['model']
    lda_fixed_model = results['lda_fixed']['model']
    
    # размеры тем для всех моделей
    lda_opt_topics = lda_opt_model.get_dominant_topics()
    bertopic_topics = bertopic_opt_model.get_document_topics()
    lda_fixed_topics = lda_fixed_model.get_dominant_topics()
    
    viz.plot_topic_sizes_comparison(lda_opt_topics, bertopic_topics, lda_fixed_topics)
    
    # 4. UMAP визуализация
    print("🗺️  UMAP визуализация...")
    viz.plot_embeddings_comparison(embeddings, lda_opt_topics, bertopic_topics, lda_fixed_topics)
    
    # 5. Анализ академического обоснования
    print("📚 Создание отчета по метрикам...")
    academic_info = results['academic_justification']
    viz.create_metrics_justification_report(academic_info, save_path)
    
    print("✅ Визуализации созданы")


def main():
    """Основная функция - ИДЕАЛЬНЫЙ ПЛАН"""
    parser = argparse.ArgumentParser(description='Тематическое моделирование: ИДЕАЛЬНЫЙ ПЛАН')
    parser.add_argument('--data', default='data_for_analysis.csv', help='путь к исходным данным')
    parser.add_argument('--skip-preprocessing', action='store_true', help='пропустить предобработку')
    parser.add_argument('--skip-embeddings', action='store_true', help='пропустить создание эмбеддингов')
    parser.add_argument('--skip-viz', action='store_true', help='пропустить визуализации')
    
    args = parser.parse_args()
    
    print("🚀 ЗАПУСК ПАЙПЛАЙНА: ИДЕАЛЬНЫЙ ПЛАН")
    print("=" * 60)
    print("📋 План:")
    print("   1. Единая система композитных метрик для LDA и BERTopic")
    print("   2. Сохранение эмбеддингов для ускорения")
    print("   3. Дополнительное сравнение LDA с фиксированным количеством тем")
    print("   4. Академическое обоснование выбора метрик")
    print("=" * 60)
    
    # Убедимся, что все папки для результатов существуют
    ensure_dir(RESULTS_PATH)
    ensure_dir(FIGURES_PATH)
    ensure_dir(TABLES_PATH)

    # --- Шаг 1: Загрузка и подготовка данных ---
    print("\n--- Шаг 1: Загрузка и подготовка данных ---")
    if os.path.exists(EMBEDDINGS_CACHE):
        print(f"Кэш эмбеддингов найден. Загрузка из {EMBEDDINGS_CACHE}...")
        with open(EMBEDDINGS_CACHE, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        print(f"Кэш эмбеддингов не найден. Загрузка и обработка данных из {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        df_filtered = df[df['text'].str.len() > 50].copy()
        
        print("Лемматизация текста...")
        preprocessor = TextPreprocessor()
        processed_texts = preprocessor.preprocess_corpus(df_filtered['text'].tolist())
        
        print("Создание эмбеддингов...")
        embeddings = load_or_create_embeddings(df_filtered['text'].tolist(), EMBEDDINGS_CACHE)

    # --- Шаг 2: Обучение и оценка моделей ---
    print("\n--- Шаг 2: Обучение и оценка моделей ---")
    lda_results = find_optimal_topic_model(LDAModel, processed_texts, embeddings, range(10, 36, 5))
    bertopic_results = find_optimal_topic_model(BERTopicModel, df_filtered, embeddings, [5, 10, 15, 20, 25])
    
    optimal_lda_params = lda_results['best_params']
    best_lda_score = lda_results['best_score']
    optimal_bertopic_params = bertopic_results['best_params']
    best_bertopic_score = bertopic_results['best_score']
    
    print(f"Оптимальная модель LDA: {optimal_lda_params} с композитной оценкой {best_lda_score:.3f}")
    print(f"Оптимальная модель BERTopic: {optimal_bertopic_params} с композитной оценкой {best_bertopic_score:.3f}")
    
    # Сохраняем детальные результаты в pkl
    with open(os.path.join(RESULTS_PATH, 'detailed_evaluation_results.pkl'), 'wb') as f:
        pickle.dump({
            'lda_results': lda_results,
            'bertopic_results': bertopic_results
        }, f)

    # --- Шаг 3: Комплексная оценка ---
    print("\n--- Шаг 3: Комплексная оценка ---")
    lda_model = LDAModel(random_state=42)
    lda_model.prepare_corpus(processed_texts)
    lda_model.train(num_topics=optimal_lda_params['num_topics'])
    
    bertopic_model = BERTopicModel(random_state=42)
    bertopic_model.train(
        df_filtered['text'].tolist(),
        min_cluster_size=optimal_bertopic_params['min_cluster_size'],
        nr_topics="auto",
        calculate_probabilities=True,
        embeddings=embeddings
    )
    
    # --- Шаг 4: Справедливое сравнение с фиксированным количеством тем ---
    print("\n--- Шаг 4: Справедливое сравнение ---")
    fair_lda_model_params = {'num_topics': optimal_bertopic_params['n_topics']}
    fair_lda_model = LDAModel(random_state=42)
    fair_lda_model.prepare_corpus(processed_texts)
    fair_lda_model.train(num_topics=fair_lda_model_params['num_topics'])
    
    # --- Шаг 5: Визуализация и сохранение результатов ---
    print("\n--- Шаг 5: Визуализация и сохранение ---")
    
    # Создание и сохранение сравнительной таблицы
    comparison_df = pd.DataFrame([
        {
            'Model': 'LDA',
            'Topics': optimal_lda_params['num_topics'],
            'Composite Score': best_lda_score,
            'Coherence CV': lda_results['best_metrics']['coherence_cv'],
            'Silhouette': lda_results['best_metrics']['silhouette'],
            'Coverage': lda_results['best_metrics']['coverage']
        },
        {
            'Model': 'BERTopic',
            'Topics': optimal_bertopic_params['n_topics'],
            'Composite Score': best_bertopic_score,
            'Coherence CV': bertopic_results['best_metrics']['coherence_cv'],
            'Silhouette': bertopic_results['best_metrics']['silhouette'],
            'Coverage': bertopic_results['best_metrics']['coverage']
        },
        {
            'Model': 'Fair LDA',
            'Topics': fair_lda_model_params['num_topics'],
            'Composite Score': fair_lda_score,
            'Coherence CV': fair_lda_metrics['coherence_cv'],
            'Silhouette': fair_lda_metrics['silhouette'],
            'Coverage': fair_lda_metrics['coverage']
        }
    ])
    comparison_df.to_csv(os.path.join(TABLES_PATH, 'model_comparison.csv'), index=False)
    
    print("Сравнительная таблица:")
    print(comparison_df.to_markdown(index=False))
    
    # Визуализация
    visualize_embeddings_comparison(lda_model, bertopic_model, embeddings, df_filtered, os.path.join(FIGURES_PATH, 'embeddings_comparison.png'))
    visualize_metrics_comparison(lda_results, bertopic_results, os.path.join(FIGURES_PATH, 'metrics_comparison.png'))
    visualize_comprehensive_comparison(comparison_df, os.path.join(FIGURES_PATH, 'comprehensive_comparison.png'))
    
    # Сохранение примеров тем для отчета
    lda_topics = lda_model.show_topics(num_topics=optimal_lda_params['num_topics'], num_words=10, formatted=False)
    bertopic_topics = bertopic_model.get_topic_info()
    
    topic_examples = {
        "LDA": {f"Topic {t[0]}": [w[0] for w in t[1]] for t in lda_topics},
        "BERTopic": {f"Topic {row.Topic}": row.Name for _, row in bertopic_topics.iterrows() if row.Topic != -1}
    }
    
    # Преобразуем в DataFrame для удобства
    lda_topic_df = pd.DataFrame([f"Тема {t[0]}: {', '.join([w[0] for w in t[1][:7]])}" for t in lda_topics], columns=['LDA Topics'])
    bertopic_topic_df = pd.DataFrame([f"Тема {row.Topic}: {row.Name}" for _, row in bertopic_topics.iterrows() if row.Topic != -1], columns=['BERTopic Topics'])
    
    # Сохраняем примеры тем в CSV
    pd.concat([lda_topic_df, bertopic_topic_df], axis=1).to_csv(os.path.join(TABLES_PATH, 'topic_examples.csv'), index=False)


    print("\nИДЕАЛЬНЫЙ ПЛАН: ЗАВЕРШЕН")
    print("="*30)


if __name__ == '__main__':
    main() 