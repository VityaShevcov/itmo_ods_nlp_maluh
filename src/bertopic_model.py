import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import pickle


class BERTopicModel:
    """класс для работы с BERTopic моделью"""
    
    def __init__(self, 
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 random_state: int = 42):
        self.random_state = random_state
        self.embedding_model_name = embedding_model
        self.model = None
        self.embeddings = None
        self.documents = None
        
        # инициализируем компоненты
        self._setup_components()
    
    def _setup_components(self):
        """настройка компонентов BERTopic"""
        # модель эмбеддингов
        self.sentence_model = SentenceTransformer(self.embedding_model_name)
        
        # снижение размерности
        self.umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=self.random_state
        )
        
        # кластеризация (параметры будут настраиваться)
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=15,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # векторизация для c-TF-IDF
        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words=None,  # уже удалили в предобработке
            min_df=2
        )
    
    def train(self, documents: List[str], 
              min_cluster_size: int = 15,
              nr_topics: Optional[Union[int, str]] = None,
              calculate_probabilities: bool = True,
              embeddings: Optional[np.ndarray] = None) -> None:
        """обучение BERTopic модели"""
        self.documents = documents
        
        # обновляем параметр кластеризации
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # создаем модель
        self.model = BERTopic(
            embedding_model=self.sentence_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            calculate_probabilities=calculate_probabilities,
            verbose=True
        )
        
        # используем готовые эмбеддинги или создаем новые
        if embeddings is not None:
            print("📂 Используем готовые эмбеддинги...")
            self.embeddings = embeddings
        else:
            print("🔄 Создание эмбеддингов...")
            self.embeddings = self.sentence_model.encode(documents, show_progress_bar=True)
        
        print("🤖 Обучение BERTopic...")
        topics, probabilities = self.model.fit_transform(documents, self.embeddings)
        
        # объединяем похожие темы если нужно
        if nr_topics is not None and nr_topics != "auto":
            self.model.reduce_topics(documents, nr_topics=nr_topics)
        elif nr_topics == "auto":
            self.model.reduce_topics(documents, nr_topics="auto")
        
        print(f"✅ Найдено {len(self.model.get_topic_info())} тем")
        print(f"📊 Количество outliers: {sum(1 for t in topics if t == -1)}")
    
    def get_topics(self, num_words: int = 10) -> List[List[Tuple[str, float]]]:
        """получение топ-слов для каждой темы"""
        if self.model is None:
            raise ValueError("модель не обучена")
        
        topics = []
        topic_info = self.model.get_topic_info()
        
        # Сортируем темы по размеру (исключая outliers)
        valid_topics = topic_info[topic_info['Topic'] != -1].sort_values('Count', ascending=False)
        
        for _, row in valid_topics.iterrows():
            topic_id = row['Topic']
            try:
                topic_words = self.model.get_topic(topic_id)
                if topic_words:  # проверяем что тема не пустая
                    # Берем только нужное количество слов
                    topic_words_limited = topic_words[:num_words]
                    topics.append(topic_words_limited)
            except Exception as e:
                print(f"Ошибка при получении темы {topic_id}: {e}")
                continue
        
        return topics
    
    def get_document_topics(self) -> List[int]:
        """получение темы для каждого документа"""
        if self.model is None:
            raise ValueError("модель не обучена")
        
        topics, _ = self.model.transform(self.documents, self.embeddings)
        return topics
    
    def get_topic_info(self) -> pd.DataFrame:
        """получение информации о темах"""
        if self.model is None:
            raise ValueError("модель не обучена")
        
        return self.model.get_topic_info()
    
    def get_representative_docs(self, topic_id: int, num_docs: int = 5) -> List[str]:
        """получение характерных документов для темы"""
        if self.model is None:
            raise ValueError("модель не обучена")
        
        representative_docs = self.model.get_representative_docs(topic_id)
        return representative_docs[:num_docs]
    
    def save_model(self, filepath: str) -> None:
        """сохранение модели"""
        if self.model is None:
            raise ValueError("модель не обучена")
        
        # сохраняем модель и дополнительные данные
        model_data = {
            'model': self.model,
            'embeddings': self.embeddings,
            'documents': self.documents,
            'embedding_model_name': self.embedding_model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"модель сохранена в {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """загрузка модели"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.embeddings = model_data['embeddings']
        self.documents = model_data['documents']
        self.embedding_model_name = model_data['embedding_model_name']
        
        print(f"модель загружена из {filepath}")


def tune_bertopic_clustering(documents: List[str],
                            min_cluster_sizes: List[int],
                            embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                            random_state: int = 42) -> pd.DataFrame:
    """подбор оптимального min_cluster_size для BERTopic"""
    
    # Подготавливаем токенизированные тексты для coherence
    from src.data_preprocessing import TextPreprocessor
    preprocessor = TextPreprocessor()
    processed_texts = []
    for doc in documents:
        tokens = preprocessor.preprocess_text(doc)
        if tokens:  # только непустые документы
            processed_texts.append(tokens)
    
    results = []
    
    for min_cluster_size in min_cluster_sizes:
        print(f"тестируем min_cluster_size={min_cluster_size}...")
        
        # создаем и обучаем модель
        bertopic = BERTopicModel(
            embedding_model=embedding_model,
            random_state=random_state
        )
        
        bertopic.train(
            documents=documents,
            min_cluster_size=min_cluster_size,
            calculate_probabilities=True
        )
        
        # собираем метрики
        topic_info = bertopic.get_topic_info()
        num_topics = len(topic_info) - 1 if -1 in topic_info['Topic'].values else len(topic_info)
        num_outliers = topic_info[topic_info['Topic'] == -1]['Count'].sum() if -1 in topic_info['Topic'].values else 0
        total_docs = len(documents)
        
        # средний размер темы
        if num_topics > 0:
            topic_sizes = topic_info[topic_info['Topic'] != -1]['Count']
            avg_topic_size = topic_sizes.mean()
            min_topic_size = topic_sizes.min()
            max_topic_size = topic_sizes.max()
        else:
            avg_topic_size = 0
            min_topic_size = 0
            max_topic_size = 0
        
        # расчет coherence_cv
        coherence_cv = 0.0
        try:
            topics = bertopic.get_topics()
            if topics and processed_texts:
                from src.evaluation import TopicEvaluator
                evaluator = TopicEvaluator()
                topic_words = []
                for topic in topics:
                    words = [word for word, _ in topic if isinstance(word, str)]
                    if words:
                        topic_words.append(words)
                
                if topic_words:
                    coherence_cv = evaluator.compute_coherence(topic_words, processed_texts, 'c_v')
        except Exception as e:
            print(f"Ошибка при расчете coherence для min_cluster_size={min_cluster_size}: {e}")
            coherence_cv = 0.0
        
        results.append({
            'min_cluster_size': min_cluster_size,
            'num_topics': num_topics,
            'outlier_ratio': num_outliers / total_docs,
            'coverage': (total_docs - num_outliers) / total_docs,
            'avg_topic_size': avg_topic_size,
            'min_topic_size': min_topic_size,
            'max_topic_size': max_topic_size,
            'coherence_cv': coherence_cv,
            'total_docs': total_docs,
            'outliers': num_outliers
        })
        
        print(f"  найдено тем: {num_topics}")
        print(f"  outliers: {num_outliers} ({num_outliers/total_docs:.1%})")
        print(f"  coherence_cv: {coherence_cv:.3f}")
    
    return pd.DataFrame(results)


def compute_topic_diversity(topics: List[List[Tuple[str, float]]], 
                           top_k: int = 10) -> float:
    """вычисление разнообразия тем (доля уникальных слов)"""
    all_words = set()
    topic_words = set()
    
    for topic in topics:
        words = [word for word, _ in topic[:top_k]]
        topic_words.update(words)
        all_words.update(words)
    
    if len(all_words) == 0:
        return 0.0
    
    return len(topic_words) / len(all_words) 