import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from umap import UMAP


class TopicVisualizer:
    """класс для визуализации результатов тематического моделирования"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_coherence_scores(self, 
                             tuning_results: pd.DataFrame,
                             title: str = "Coherence vs Number of Topics") -> None:
        """график кохерентности по числу тем для LDA"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(tuning_results['num_topics'], tuning_results['coherence_cv'], 
                'o-', label='C_V Coherence', linewidth=2, markersize=6)
        
        if 'coherence_umass' in tuning_results.columns:
            ax2 = ax.twinx()
            ax2.plot(tuning_results['num_topics'], tuning_results['coherence_umass'], 
                    's-', color='red', label='UMass Coherence', linewidth=2, markersize=6)
            ax2.set_ylabel('UMass Coherence', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_xlabel('Number of Topics')
        ax.set_ylabel('C_V Coherence')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # отмечаем оптимальное число тем
        best_idx = tuning_results['coherence_cv'].idxmax()
        best_topics = tuning_results.loc[best_idx, 'num_topics']
        best_score = tuning_results.loc[best_idx, 'coherence_cv']
        
        ax.axvline(x=best_topics, color='green', linestyle='--', alpha=0.7)
        ax.annotate(f'Optimal: {best_topics} topics\nC_V: {best_score:.3f}',
                   xy=(best_topics, best_score), xytext=(10, 10),
                   textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def plot_bertopic_tuning(self, 
                            tuning_results: pd.DataFrame,
                            title: str = "BERTopic Parameter Tuning") -> None:
        """график подбора параметров для BERTopic"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # число тем
        ax1.plot(tuning_results['min_cluster_size'], tuning_results['num_topics'], 'o-')
        ax1.set_xlabel('Min Cluster Size')
        ax1.set_ylabel('Number of Topics')
        ax1.set_title('Topics vs Min Cluster Size')
        ax1.grid(True, alpha=0.3)
        
        # доля outliers
        ax2.plot(tuning_results['min_cluster_size'], tuning_results['outlier_ratio'], 'o-', color='red')
        ax2.set_xlabel('Min Cluster Size')
        ax2.set_ylabel('Outlier Ratio')
        ax2.set_title('Outliers vs Min Cluster Size')
        ax2.grid(True, alpha=0.3)
        
        # средний размер темы
        ax3.plot(tuning_results['min_cluster_size'], tuning_results['avg_topic_size'], 'o-', color='green')
        ax3.set_xlabel('Min Cluster Size')
        ax3.set_ylabel('Average Topic Size')
        ax3.set_title('Average Topic Size vs Min Cluster Size')
        ax3.grid(True, alpha=0.3)
        
        # общий график
        ax4.bar(tuning_results['min_cluster_size'].astype(str), tuning_results['num_topics'], alpha=0.7)
        ax4.set_xlabel('Min Cluster Size')
        ax4.set_ylabel('Number of Topics')
        ax4.set_title('Topic Count Distribution')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self, 
                               comparison_df: pd.DataFrame,
                               title: str = "Model Comparison") -> None:
        """сравнение метрик моделей"""
        metrics_to_plot = ['coherence_cv', 'coherence_umass', 'silhouette', 'topic_diversity']
        
        # фильтруем только нужные метрики
        plot_df = comparison_df[comparison_df['metric'].isin(metrics_to_plot)].copy()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(plot_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, plot_df['LDA'], width, label='LDA', alpha=0.8)
        bars2 = ax.bar(x + width/2, plot_df['BERTopic'], width, label='BERTopic', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['metric'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # добавляем значения на столбцы
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
                   f'{height1:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01,
                   f'{height2:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_topic_sizes(self, 
                        lda_topics: List[int], 
                        bertopic_topics: List[int],
                        title: str = "Topic Size Distribution") -> None:
        """распределение размеров тем"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # LDA
        lda_counts = pd.Series(lda_topics).value_counts().sort_index()
        lda_counts = lda_counts[lda_counts.index != -1]  # исключаем outliers
        
        ax1.bar(range(len(lda_counts)), lda_counts.values, alpha=0.7)
        ax1.set_xlabel('Topic ID')
        ax1.set_ylabel('Number of Documents')
        ax1.set_title('LDA Topic Sizes')
        ax1.grid(True, alpha=0.3)
        
        # BERTopic
        bertopic_counts = pd.Series(bertopic_topics).value_counts().sort_values(ascending=False)
        bertopic_counts = bertopic_counts[bertopic_counts.index != -1]  # исключаем outliers
        
        ax2.bar(range(len(bertopic_counts)), bertopic_counts.values, alpha=0.7, color='orange')
        ax2.set_xlabel('Topic ID (sorted by size)')
        ax2.set_ylabel('Number of Documents')
        ax2.set_title('BERTopic Topic Sizes')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_embeddings_2d(self, 
                          embeddings: np.ndarray,
                          labels: List[int],
                          method: str = 'umap',
                          title: str = "Document Embeddings") -> None:
        """2d визуализация эмбеддингов документов"""
        
        # снижение размерности
        if method == 'umap':
            reducer = UMAP(n_components=2, random_state=42)
        else:  # pca
            reducer = PCA(n_components=2, random_state=42)
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # создаем dataframe для plotly
        df_plot = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'topic': [str(label) if label != -1 else 'Outlier' for label in labels]
        })
        
        # plotly график
        fig = px.scatter(df_plot, x='x', y='y', color='topic',
                        title=f"{title} ({method.upper()})",
                        hover_data=['topic'])
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        fig.show()
    
    def create_wordcloud(self, 
                        topic_words: List[Tuple[str, float]],
                        title: str = "Topic WordCloud") -> None:
        """создание облака слов для темы"""
        
        # подготавливаем словарь слов и их весов
        word_freq = {word: weight for word, weight in topic_words}
        
        # создаем облако слов
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=50,
            colormap='viridis',
            font_path=None  # для русского языка может потребоваться путь к шрифту
        ).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_topic_words_comparison(self, 
                                   lda_topic: List[Tuple[str, float]],
                                   bertopic_topic: List[Tuple[str, float]],
                                   lda_name: str = "LDA Topic",
                                   bertopic_name: str = "BERTopic Topic") -> None:
        """сравнение топ-слов двух тем"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # LDA
        words1, scores1 = zip(*lda_topic[:10])
        y_pos1 = np.arange(len(words1))
        
        ax1.barh(y_pos1, scores1, alpha=0.7)
        ax1.set_yticks(y_pos1)
        ax1.set_yticklabels(words1)
        ax1.invert_yaxis()
        ax1.set_xlabel('Score')
        ax1.set_title(lda_name)
        ax1.grid(True, alpha=0.3)
        
        # BERTopic
        words2, scores2 = zip(*bertopic_topic[:10])
        y_pos2 = np.arange(len(words2))
        
        ax2.barh(y_pos2, scores2, alpha=0.7, color='orange')
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(words2)
        ax2.invert_yaxis()
        ax2.set_xlabel('Score')
        ax2.set_title(bertopic_name)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_source_distribution(self, 
                                df: pd.DataFrame,
                                lda_topics: List[int],
                                bertopic_topics: List[int]) -> None:
        """распределение источников по темам"""
        
        # подготовка данных
        df_plot = df.copy()
        df_plot['lda_topic'] = lda_topics
        df_plot['bertopic_topic'] = bertopic_topics
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # LDA
        lda_source_dist = df_plot.groupby(['lda_topic', 'source']).size().unstack(fill_value=0)
        lda_source_dist = lda_source_dist.loc[lda_source_dist.index != -1]  # исключаем outliers
        
        lda_source_dist.plot(kind='bar', stacked=True, ax=ax1, alpha=0.8)
        ax1.set_title('LDA: Source Distribution by Topic')
        ax1.set_xlabel('Topic')
        ax1.set_ylabel('Number of Documents')
        ax1.legend(title='Source')
        ax1.grid(True, alpha=0.3)
        
        # BERTopic  
        bertopic_source_dist = df_plot.groupby(['bertopic_topic', 'source']).size().unstack(fill_value=0)
        bertopic_source_dist = bertopic_source_dist.loc[bertopic_source_dist.index != -1]
        
        bertopic_source_dist.plot(kind='bar', stacked=True, ax=ax2, alpha=0.8)
        ax2.set_title('BERTopic: Source Distribution by Topic')
        ax2.set_xlabel('Topic')
        ax2.set_ylabel('Number of Documents')
        ax2.legend(title='Source')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_all_figures(self, filepath_prefix: str) -> None:
        """сохранение всех созданных графиков"""
        # эта функция будет вызываться после создания всех графиков
        # для сохранения в файлы для отчета
        pass


def create_topic_summary_table(lda_topics: List[List[Tuple[str, float]]],
                              bertopic_topics: List[List[Tuple[str, float]]],
                              num_topics: int = 5) -> pd.DataFrame:
    """создание таблицы сравнения тем"""
    
    summary_data = []
    
    for i in range(min(num_topics, len(lda_topics), len(bertopic_topics))):
        lda_words = ', '.join([word for word, _ in lda_topics[i][:10]])
        bertopic_words = ', '.join([word for word, _ in bertopic_topics[i][:10]])
        
        summary_data.append({
            'Topic_ID': i,
            'LDA_Words': lda_words,
            'BERTopic_Words': bertopic_words
        })
    
    return pd.DataFrame(summary_data) 