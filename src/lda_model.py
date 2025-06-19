import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import gensim
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel
import pickle


class LDAModel:
    """класс для работы с LDA моделью"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.dictionary = None
        self.corpus = None
        self.texts = None
        
    def prepare_corpus(self, texts: List[List[str]], 
                      no_below: int = 5, no_above: float = 0.5) -> None:
        """подготовка корпуса для LDA"""
        self.texts = texts
        
        # создаем словарь
        self.dictionary = corpora.Dictionary(texts)
        
        # фильтруем экстремальные частоты
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        
        # создаем корпус (bag of words)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        print(f"словарь содержит {len(self.dictionary)} уникальных токенов")
        print(f"корпус содержит {len(self.corpus)} документов")
    
    def train(self, num_topics: int, alpha: str = 'symmetric', 
              beta: str = 'auto', iterations: int = 100,
              passes: int = 10, workers: int = 4) -> None:
        """обучение LDA модели"""
        if self.corpus is None:
            raise ValueError("корпус не подготовлен, вызовите prepare_corpus")
        
        self.model = LdaMulticore(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=self.random_state,
            alpha=alpha,
            eta=beta,
            iterations=iterations,
            passes=passes,
            workers=workers,
            eval_every=None
        )
        
        print(f"обучена LDA модель с {num_topics} темами")
    
    def get_topics(self, num_words: int = 10) -> List[List[Tuple[str, float]]]:
        """получение топ-слов для каждой темы"""
        if self.model is None:
            raise ValueError("модель не обучена")
        
        topics = []
        for topic_id in range(self.model.num_topics):
            topic_words = self.model.show_topic(topic_id, topn=num_words)
            topics.append(topic_words)
        
        return topics
    
    def get_document_topics(self, minimum_probability: float = 0.01) -> List[List[Tuple[int, float]]]:
        """получение распределения тем для документов"""
        if self.model is None or self.corpus is None:
            raise ValueError("модель или корпус не готовы")
        
        doc_topics = []
        for doc in self.corpus:
            topics = self.model.get_document_topics(doc, minimum_probability=minimum_probability)
            doc_topics.append(topics)
        
        return doc_topics
    
    def get_dominant_topics(self) -> List[int]:
        """получение доминирующей темы для каждого документа"""
        doc_topics = self.get_document_topics()
        dominant_topics = []
        
        for topics in doc_topics:
            if topics:
                # находим тему с максимальной вероятностью
                dominant_topic = max(topics, key=lambda x: x[1])[0]
            else:
                dominant_topic = -1  # нет доминирующей темы
            dominant_topics.append(dominant_topic)
        
        return dominant_topics
    
    def compute_perplexity(self, test_corpus: Optional[List] = None) -> float:
        """вычисление перплексии"""
        if self.model is None:
            raise ValueError("модель не обучена")
        
        corpus_to_use = test_corpus if test_corpus is not None else self.corpus
        perplexity = self.model.log_perplexity(corpus_to_use)
        return np.exp(-perplexity)
    
    def save_model(self, filepath: str) -> None:
        """сохранение модели"""
        if self.model is None:
            raise ValueError("модель не обучена")
        
        # сохраняем модель, словарь и корпус
        model_data = {
            'model': self.model,
            'dictionary': self.dictionary,
            'corpus': self.corpus,
            'texts': self.texts
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"модель сохранена в {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """загрузка модели"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.dictionary = model_data['dictionary']
        self.corpus = model_data['corpus']
        self.texts = model_data['texts']
        
        print(f"модель загружена из {filepath}")


def tune_lda_topics(texts: List[List[str]], 
                   topic_range: List[int],
                   test_size: float = 0.2,
                   random_state: int = 42) -> pd.DataFrame:
    """подбор оптимального числа тем для LDA"""
    
    # разделяем на train/test для перплексии
    np.random.seed(random_state)
    n_docs = len(texts)
    test_indices = np.random.choice(n_docs, size=int(n_docs * test_size), replace=False)
    train_indices = np.setdiff1d(range(n_docs), test_indices)
    
    train_texts = [texts[i] for i in train_indices]
    test_texts = [texts[i] for i in test_indices]
    
    results = []
    
    for num_topics in topic_range:
        print(f"тестируем {num_topics} тем...")
        
        # создаем и обучаем модель
        lda = LDAModel(random_state=random_state)
        lda.prepare_corpus(train_texts)
        lda.train(num_topics=num_topics)
        
        # test корпус
        test_corpus = [lda.dictionary.doc2bow(text) for text in test_texts]
        
        # метрики
        coherence_cv = compute_coherence(lda.model, train_texts, lda.dictionary, 'c_v')
        coherence_umass = compute_coherence(lda.model, train_texts, lda.dictionary, 'u_mass')
        perplexity = lda.compute_perplexity(test_corpus)
        
        results.append({
            'num_topics': num_topics,
            'coherence_cv': coherence_cv,
            'coherence_umass': coherence_umass,
            'perplexity': perplexity
        })
    
    return pd.DataFrame(results)


def compute_coherence(model, texts: List[List[str]], 
                     dictionary, measure: str = 'c_v') -> float:
    """вычисление кохерентности модели"""
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        coherence=measure
    )
    
    return coherence_model.get_coherence() 