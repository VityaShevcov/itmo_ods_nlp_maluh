#!/usr/bin/env python3
"""
Основной скрипт для запуска пайплайна тематического моделирования
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# добавляем путь к нашим модулям
sys.path.append('src')

from data_preprocessing import load_and_clean_data, TextPreprocessor
from lda_model import LDAModel, tune_lda_topics
from bertopic_model import BERTopicModel, tune_bertopic_clustering
from evaluation import TopicEvaluator, compare_models
from visualization import TopicVisualizer


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


def run_lda_modeling(processed_texts: List[List[str]], 
                    save_path: str = 'models') -> Dict[str, Any]:
    """запуск LDA моделирования"""
    print("\n=== ЭТАП 2: LDA моделирование ===")
    
    os.makedirs(save_path, exist_ok=True)
    
    # подбор числа тем
    print("подбор оптимального числа тем...")
    topic_range = list(range(5, 31, 5))  # от 5 до 30 с шагом 5
    tuning_results = tune_lda_topics(processed_texts, topic_range)
    
    # находим оптимальное число тем
    best_topics = tuning_results.loc[tuning_results['coherence_cv'].idxmax(), 'num_topics']
    print(f"оптимальное число тем: {best_topics}")
    
    # обучаем финальную модель
    print("обучение финальной LDA модели...")
    lda_model = LDAModel(random_state=42)
    lda_model.prepare_corpus(processed_texts)
    lda_model.train(num_topics=int(best_topics))
    
    # сохраняем модель
    lda_model.save_model(f'{save_path}/lda_model.pkl')
    tuning_results.to_csv(f'{save_path}/lda_tuning_results.csv', index=False)
    
    return {
        'model': lda_model,
        'tuning_results': tuning_results,
        'best_topics': best_topics
    }


def run_bertopic_modeling(df: pd.DataFrame, 
                         save_path: str = 'models') -> Dict[str, Any]:
    """запуск BERTopic моделирования"""
    print("\n=== ЭТАП 3: BERTopic моделирование ===")
    
    # объединяем тексты обратно в строки для BERTopic
    documents = df['text'].tolist()
    
    # подбор параметров кластеризации
    print("подбор параметров кластеризации...")
    cluster_sizes = [10, 15, 20, 30]
    tuning_results = tune_bertopic_clustering(documents, cluster_sizes)
    
    # выбираем оптимальный параметр (баланс числа тем и outliers)
    tuning_results['score'] = tuning_results['num_topics'] * (1 - tuning_results['outlier_ratio'])
    best_cluster_size = tuning_results.loc[tuning_results['score'].idxmax(), 'min_cluster_size']
    print(f"оптимальный min_cluster_size: {best_cluster_size}")
    
    # обучаем финальную модель
    print("обучение финальной BERTopic модели...")
    bertopic_model = BERTopicModel(random_state=42)
    bertopic_model.train(documents, min_cluster_size=int(best_cluster_size))
    
    # сохраняем модель
    bertopic_model.save_model(f'{save_path}/bertopic_model.pkl')
    tuning_results.to_csv(f'{save_path}/bertopic_tuning_results.csv', index=False)
    
    return {
        'model': bertopic_model,
        'tuning_results': tuning_results,
        'best_cluster_size': best_cluster_size
    }


def run_evaluation(lda_results: Dict[str, Any], 
                  bertopic_results: Dict[str, Any],
                  processed_texts: List[List[str]],
                  df: pd.DataFrame,
                  save_path: str = 'report') -> Dict[str, Any]:
    """запуск оценки и сравнения моделей"""
    print("\n=== ЭТАП 4: Оценка и сравнение моделей ===")
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f'{save_path}/figures', exist_ok=True)
    
    evaluator = TopicEvaluator()
    
    # получаем результаты моделей
    lda_model = lda_results['model']
    bertopic_model = bertopic_results['model']
    
    # темы и метки документов
    lda_topics = lda_model.get_topics()
    lda_doc_topics = lda_model.get_dominant_topics()
    
    bertopic_topics = bertopic_model.get_topics()
    bertopic_doc_topics = bertopic_model.get_document_topics()
    
    # эмбеддинги для silhouette
    embeddings = bertopic_model.embeddings
    
    # оценка LDA
    print("оценка LDA модели...")
    lda_eval = evaluator.evaluate_model(
        topics=lda_topics,
        texts=processed_texts,
        embeddings=embeddings,
        doc_topics=lda_doc_topics,
        model_name="LDA"
    )
    
    # оценка BERTopic
    print("оценка BERTopic модели...")
    bertopic_eval = evaluator.evaluate_model(
        topics=bertopic_topics,
        texts=processed_texts,
        embeddings=embeddings,
        doc_topics=bertopic_doc_topics,
        model_name="BERTopic"
    )
    
    # сравнение моделей
    comparison_df = compare_models(lda_eval, bertopic_eval)
    
    # сохраняем результаты
    comparison_df.to_csv(f'{save_path}/model_comparison.csv', index=False)
    
    with open(f'{save_path}/evaluation_results.pkl', 'wb') as f:
        pickle.dump({
            'lda_eval': lda_eval,
            'bertopic_eval': bertopic_eval,
            'comparison': comparison_df
        }, f)
    
    print("оценка завершена")
    return {
        'lda_eval': lda_eval,
        'bertopic_eval': bertopic_eval,
        'comparison': comparison_df
    }


def create_visualizations(lda_results: Dict[str, Any], 
                         bertopic_results: Dict[str, Any],
                         evaluation_results: Dict[str, Any],
                         df: pd.DataFrame,
                         save_path: str = 'report/figures') -> None:
    """создание всех визуализаций"""
    print("\n=== ЭТАП 5: Создание визуализаций ===")
    
    viz = TopicVisualizer()
    
    # графики подбора параметров
    print("графики подбора параметров...")
    viz.plot_coherence_scores(lda_results['tuning_results'])
    viz.plot_bertopic_tuning(bertopic_results['tuning_results'])
    
    # сравнение метрик
    print("сравнение метрик моделей...")
    viz.plot_metrics_comparison(evaluation_results['comparison'])
    
    # размеры тем
    lda_doc_topics = lda_results['model'].get_dominant_topics()
    bertopic_doc_topics = bertopic_results['model'].get_document_topics()
    viz.plot_topic_sizes(lda_doc_topics, bertopic_doc_topics)
    
    # распределение источников по темам
    viz.plot_source_distribution(df, lda_doc_topics, bertopic_doc_topics)
    
    print("визуализации созданы")


def main():
    """основная функция"""
    parser = argparse.ArgumentParser(description='Тематическое моделирование отзывов')
    parser.add_argument('--data', default='data_for_analysis.csv', help='путь к исходным данным')
    parser.add_argument('--skip-preprocessing', action='store_true', help='пропустить предобработку')
    parser.add_argument('--skip-lda', action='store_true', help='пропустить LDA')
    parser.add_argument('--skip-bertopic', action='store_true', help='пропустить BERTopic')
    parser.add_argument('--skip-viz', action='store_true', help='пропустить визуализации')
    
    args = parser.parse_args()
    
    print("🚀 Запуск пайплайна тематического моделирования")
    print("=" * 50)
    
    # предобработка
    if not args.skip_preprocessing:
        df, processed_texts = run_preprocessing(args.data)
    else:
        df = pd.read_csv('data/processed/processed_reviews.csv')
        with open('data/processed/processed_texts.pkl', 'rb') as f:
            processed_texts = pickle.load(f)
    
    # LDA моделирование
    if not args.skip_lda:
        lda_results = run_lda_modeling(processed_texts)
    else:
        lda_model = LDAModel()
        lda_model.load_model('models/lda_model.pkl')
        lda_tuning = pd.read_csv('models/lda_tuning_results.csv')
        lda_results = {'model': lda_model, 'tuning_results': lda_tuning}
    
    # BERTopic моделирование
    if not args.skip_bertopic:
        bertopic_results = run_bertopic_modeling(df)
    else:
        bertopic_model = BERTopicModel()
        bertopic_model.load_model('models/bertopic_model.pkl')
        bertopic_tuning = pd.read_csv('models/bertopic_tuning_results.csv')
        bertopic_results = {'model': bertopic_model, 'tuning_results': bertopic_tuning}
    
    # оценка моделей
    evaluation_results = run_evaluation(lda_results, bertopic_results, processed_texts, df)
    
    # визуализации
    if not args.skip_viz:
        create_visualizations(lda_results, bertopic_results, evaluation_results, df)
    
    print("\n✅ Пайплайн успешно завершен!")
    print(f"📊 Результаты сохранены в папках: models/, report/")
    print(f"📈 Сравнение моделей: report/model_comparison.csv")


if __name__ == '__main__':
    main() 