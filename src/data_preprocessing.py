import re
import pandas as pd
import numpy as np
from typing import List, Optional
import pymorphy2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """класс для предобработки русского текста"""
    
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self._download_nltk_data()
        
        # стоп-слова для русского языка
        self.stop_words = set(stopwords.words('russian'))
        # добавляем дополнительные стоп-слова
        additional_stops = {
            'это', 'день', 'время', 'человек', 'год', 'месяц', 'неделя',
            'раз', 'больше', 'меньше', 'хорошо', 'плохо', 'очень', 'много',
            'мало', 'также', 'тоже', 'ещё', 'еще', 'всё', 'все', 'или'
        }
        self.stop_words.update(additional_stops)
    
    def _download_nltk_data(self):
        """загружаем необходимые данные nltk"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def clean_text(self, text: str) -> str:
        """очистка текста от шума"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # удаляем html теги
        text = re.sub(r'<[^>]+>', '', text)
        
        # удаляем ссылки
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # удаляем эмодзи
        text = re.sub(r'[^\w\s\.,!?;:\-()«»"]', '', text)
        
        # заменяем цифры токеном
        text = re.sub(r'\d+', '<NUM>', text)
        
        # убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        # приводим к нижнему регистру
        text = text.lower()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """токенизация текста"""
        if not text:
            return []
        
        # используем nltk для токенизации
        tokens = word_tokenize(text, language='russian')
        
        # фильтруем только буквенные токены и <NUM>
        tokens = [token for token in tokens 
                 if token.isalpha() or token == '<NUM>']
        
        return tokens
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """лемматизация токенов"""
        lemmatized = []
        for token in tokens:
            if token == '<NUM>':
                lemmatized.append(token)
            else:
                # получаем лемму
                parsed = self.morph.parse(token)[0]
                lemma = parsed.normal_form
                lemmatized.append(lemma)
        
        return lemmatized
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """удаление стоп-слов"""
        return [token for token in tokens 
                if token not in self.stop_words and len(token) > 2]
    
    def preprocess_text(self, text: str) -> List[str]:
        """полная предобработка текста"""
        # очистка
        clean_text = self.clean_text(text)
        
        # токенизация
        tokens = self.tokenize(clean_text)
        
        # лемматизация
        lemmatized = self.lemmatize(tokens)
        
        # удаление стоп-слов
        filtered = self.remove_stopwords(lemmatized)
        
        return filtered
    
    def preprocess_corpus(self, texts: List[str], min_length: int = 3) -> List[List[str]]:
        """предобработка корпуса текстов"""
        processed = []
        
        for text in texts:
            tokens = self.preprocess_text(text)
            
            # фильтруем слишком короткие документы
            if len(tokens) >= min_length:
                processed.append(tokens)
            else:
                processed.append([])  # пустой документ
        
        return processed


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """загрузка и первичная очистка данных"""
    # читаем данные
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # переименовываем колонки на латиницу
    if 'источник' in df.columns:
        df = df.rename(columns={
            'источник': 'source',
            'текст': 'text'
        })
    
    # убираем дубликаты
    df = df.drop_duplicates(subset=['text'])
    
    # убираем пустые тексты
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.len() > 10]  # минимум 10 символов
    
    # приводим источники к стандартному виду
    df['source'] = df['source'].str.lower().str.strip()
    df['source'] = df['source'].map({'2gis': '2GIS', 'banki': 'Banki'}).fillna(df['source'])
    
    return df.reset_index(drop=True) 