import matplotlib
matplotlib.use('Agg')
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
        plt.savefig('report/figures/coherence_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        plt.savefig('report/figures/bertopic_tuning.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        plt.savefig('report/figures/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        plt.savefig('report/figures/topic_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        
        fig.write_html(f'report/figures/embeddings_2d_{method}.html')
    
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
        plt.savefig('report/figures/wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        plt.savefig('report/figures/topic_words_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        plt.savefig('report/figures/source_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_outliers(self, 
                        df: pd.DataFrame,
                        bertopic_doc_topics: List[int],
                        title: str = "Outlier Analysis") -> None:
        """анализ выбросов BERTopic"""
        
        # добавляем темы к данным
        df_analysis = df.copy()
        df_analysis['topic'] = bertopic_doc_topics
        
        # выбросы
        outliers = df_analysis[df_analysis['topic'] == -1]
        regular_docs = df_analysis[df_analysis['topic'] != -1]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # распределение по источникам
        outlier_sources = outliers['source'].value_counts()
        regular_sources = regular_docs['source'].value_counts()
        
        sources = list(set(outlier_sources.index) | set(regular_sources.index))
        outlier_counts = [outlier_sources.get(s, 0) for s in sources]
        regular_counts = [regular_sources.get(s, 0) for s in sources]
        
        x = np.arange(len(sources))
        width = 0.35
        
        ax1.bar(x - width/2, outlier_counts, width, label='Outliers', alpha=0.7, color='red')
        ax1.bar(x + width/2, regular_counts, width, label='Regular', alpha=0.7, color='blue')
        ax1.set_xlabel('Source')
        ax1.set_ylabel('Count')
        ax1.set_title('Documents by Source')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sources)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # длина текста
        outliers['text_length'] = outliers['text'].str.len()
        regular_docs['text_length'] = regular_docs['text'].str.len()
        
        ax2.hist(outliers['text_length'], bins=30, alpha=0.7, label='Outliers', color='red')
        ax2.hist(regular_docs['text_length'], bins=30, alpha=0.7, label='Regular', color='blue')
        ax2.set_xlabel('Text Length')
        ax2.set_ylabel('Count')
        ax2.set_title('Text Length Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # доля выбросов по источникам
        source_outlier_ratio = []
        source_names = []
        for source in sources:
            source_data = df_analysis[df_analysis['source'] == source]
            outlier_ratio = (source_data['topic'] == -1).mean()
            source_outlier_ratio.append(outlier_ratio)
            source_names.append(source)
        
        ax3.bar(source_names, source_outlier_ratio, alpha=0.7, color='orange')
        ax3.set_xlabel('Source')
        ax3.set_ylabel('Outlier Ratio')
        ax3.set_title('Outlier Ratio by Source')
        ax3.grid(True, alpha=0.3)
        
        # статистика
        stats_text = f"""Общая статистика:
        Всего документов: {len(df_analysis)}
        Выбросы: {len(outliers)} ({len(outliers)/len(df_analysis):.1%})
        Обычные: {len(regular_docs)} ({len(regular_docs)/len(df_analysis):.1%})
        
        Средняя длина текста:
        Выбросы: {outliers['text_length'].mean():.0f} символов
        Обычные: {regular_docs['text_length'].mean():.0f} символов"""
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Statistics')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig('report/figures/outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_topic_examples(self,
                             df: pd.DataFrame,
                             lda_model,
                             bertopic_model,
                             num_examples: int = 2) -> None:
        """создание таблицы с примерами документов для тем"""
        
        # получаем темы и документы
        lda_topics = lda_model.get_dominant_topics()
        bertopic_topics = bertopic_model.get_document_topics()
        
        df_examples = df.copy()
        df_examples['lda_topic'] = lda_topics
        df_examples['bertopic_topic'] = bertopic_topics
        
        examples_data = []
        
        # топ-5 тем для каждой модели
        lda_topic_counts = pd.Series(lda_topics).value_counts().head(5)
        bertopic_topic_counts = pd.Series([t for t in bertopic_topics if t != -1]).value_counts().head(5)
        
        # примеры для LDA
        for topic_id in lda_topic_counts.index:
            topic_docs = df_examples[df_examples['lda_topic'] == topic_id].head(num_examples)
            topic_words = lda_model.get_topics()[topic_id][:10]
            words_str = ', '.join([word for word, _ in topic_words])
            
            for idx, row in topic_docs.iterrows():
                examples_data.append({
                    'model': 'LDA',
                    'topic_id': topic_id,
                    'topic_words': words_str,
                    'document': row['text'][:200] + '...' if len(row['text']) > 200 else row['text'],
                    'source': row['source']
                })
        
        # примеры для BERTopic
        for topic_id in bertopic_topic_counts.index:
            topic_docs = df_examples[df_examples['bertopic_topic'] == topic_id].head(num_examples)
            topic_words = bertopic_model.get_topics()
            
            # находим нужную тему
            words_str = ""
            for i, topic in enumerate(topic_words):
                if i == topic_id:  # предполагаем что индекс соответствует ID
                    words_str = ', '.join([word for word, _ in topic[:10]])
                    break
            
            for idx, row in topic_docs.iterrows():
                examples_data.append({
                    'model': 'BERTopic',
                    'topic_id': topic_id,
                    'topic_words': words_str,
                    'document': row['text'][:200] + '...' if len(row['text']) > 200 else row['text'],
                    'source': row['source']
                })
        
        # сохраняем в CSV
        examples_df = pd.DataFrame(examples_data)
        examples_df.to_csv('report/topic_examples.csv', index=False, encoding='utf-8')
        print("Примеры документов сохранены в report/topic_examples.csv")
    
    def save_all_figures(self, filepath_prefix: str) -> None:
        """сохранение всех созданных графиков"""
        # этот метод можно расширить для сохранения всех графиков
        print(f"Графики сохранены с префиксом: {filepath_prefix}")
    
    def plot_composite_metrics_tuning(self, 
                                    tuning_results: pd.DataFrame,
                                    model_name: str) -> None:
        """График подбора параметров с композитными метриками"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # определяем x-ось в зависимости от модели
        if model_name == 'LDA':
            x_col = 'num_topics'
            x_label = 'Number of Topics'
        else:  # BERTopic
            x_col = 'min_cluster_size'
            x_label = 'Min Cluster Size'
        
        x_values = tuning_results[x_col]
        
        # 1. Композитная оценка
        ax1.plot(x_values, tuning_results['composite_score'], 'o-', linewidth=2, markersize=8, color='purple')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Composite Score')
        ax1.set_title(f'{model_name}: Композитная оценка')
        ax1.grid(True, alpha=0.3)
        
        # отмечаем лучший результат
        best_idx = tuning_results['composite_score'].idxmax()
        best_x = tuning_results.loc[best_idx, x_col]
        best_score = tuning_results.loc[best_idx, 'composite_score']
        ax1.axvline(x=best_x, color='red', linestyle='--', alpha=0.7)
        ax1.annotate(f'Лучший: {best_x}\nОценка: {best_score:.3f}',
                    xy=(best_x, best_score), xytext=(10, 10),
                    textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='lightcoral', alpha=0.7))
        
        # 2. Coherence CV
        ax2.plot(x_values, tuning_results['coherence_cv'], 'o-', linewidth=2, markersize=6, color='blue')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Coherence CV')
        ax2.set_title(f'{model_name}: Семантическая связность')
        ax2.grid(True, alpha=0.3)
        
        # 3. Silhouette
        ax3.plot(x_values, tuning_results['silhouette'], 'o-', linewidth=2, markersize=6, color='green')
        ax3.set_xlabel(x_label)
        ax3.set_ylabel('Silhouette Score')
        ax3.set_title(f'{model_name}: Качество кластеризации')
        ax3.grid(True, alpha=0.3)
        
        # 4. Coverage
        ax4.plot(x_values, tuning_results['coverage'], 'o-', linewidth=2, markersize=6, color='orange')
        ax4.set_xlabel(x_label)
        ax4.set_ylabel('Coverage')
        ax4.set_title(f'{model_name}: Покрытие документов')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name}: Подбор параметров по композитным метрикам', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'report/figures/{model_name.lower()}_composite_tuning.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comprehensive_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Комплексное сравнение всех моделей"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        models = comparison_df['model'].tolist()
        
        # 1. Композитная оценка (главная метрика)
        bars1 = ax1.bar(models, comparison_df['composite_score'], 
                       color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        ax1.set_title('Композитная оценка (главная метрика)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Composite Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # добавляем значения на столбцы
        for bar, value in zip(bars1, comparison_df['composite_score']):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Coherence CV
        bars2 = ax2.bar(models, comparison_df['coherence_cv'], 
                       color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        ax2.set_title('Семантическая связность (Coherence CV)')
        ax2.set_ylabel('Coherence CV')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, comparison_df['coherence_cv']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Silhouette Score
        bars3 = ax3.bar(models, comparison_df['silhouette'], 
                       color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        ax3.set_title('Качество кластеризации (Silhouette)')
        ax3.set_ylabel('Silhouette Score')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, comparison_df['silhouette']):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Coverage
        bars4 = ax4.bar(models, comparison_df['coverage'], 
                       color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        ax4.set_title('Покрытие документов (Coverage)')
        ax4.set_ylabel('Coverage')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars4, comparison_df['coverage']):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle('КОМПЛЕКСНОЕ СРАВНЕНИЕ МОДЕЛЕЙ\n(Единая система композитных метрик)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('report/figures/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_topic_sizes_comparison(self, 
                                  lda_optimal_topics: List[int],
                                  bertopic_topics: List[int], 
                                  lda_fixed_topics: List[int]) -> None:
        """Сравнение размеров тем для всех трех моделей"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # LDA оптимальная
        lda_opt_counts = pd.Series(lda_optimal_topics).value_counts().sort_values(ascending=False)
        lda_opt_counts = lda_opt_counts[lda_opt_counts.index != -1]
        
        ax1.bar(range(len(lda_opt_counts)), lda_opt_counts.values, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Topic ID (sorted by size)')
        ax1.set_ylabel('Number of Documents')
        ax1.set_title('LDA Optimal\n(по композитным метрикам)')
        ax1.grid(True, alpha=0.3)
        
        # BERTopic
        bertopic_counts = pd.Series(bertopic_topics).value_counts().sort_values(ascending=False)
        bertopic_counts = bertopic_counts[bertopic_counts.index != -1]
        
        ax2.bar(range(len(bertopic_counts)), bertopic_counts.values, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Topic ID (sorted by size)')
        ax2.set_ylabel('Number of Documents')
        ax2.set_title('BERTopic Optimal\n(по композитным метрикам)')
        ax2.grid(True, alpha=0.3)
        
        # LDA с фиксированным количеством тем
        lda_fixed_counts = pd.Series(lda_fixed_topics).value_counts().sort_values(ascending=False)
        lda_fixed_counts = lda_fixed_counts[lda_fixed_counts.index != -1]
        
        ax3.bar(range(len(lda_fixed_counts)), lda_fixed_counts.values, alpha=0.7, color='lightgreen')
        ax3.set_xlabel('Topic ID (sorted by size)')
        ax3.set_ylabel('Number of Documents')
        ax3.set_title('LDA Fixed Topics\n(как у BERTopic)')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('СРАВНЕНИЕ РАЗМЕРОВ ТЕМ', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('report/figures/topic_sizes_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_embeddings_comparison(self, 
                                 embeddings: np.ndarray,
                                 lda_optimal_topics: List[int],
                                 bertopic_topics: List[int],
                                 lda_fixed_topics: List[int]) -> None:
        """UMAP визуализация для всех трех моделей"""
        # снижение размерности
        reducer = UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
        
        # цветовые палитры
        colors = plt.cm.Set3(np.linspace(0, 1, 20))
        
        # LDA оптимальная
        unique_topics_lda_opt = list(set(lda_optimal_topics))
        for i, topic in enumerate(unique_topics_lda_opt):
            if topic != -1:
                mask = np.array(lda_optimal_topics) == topic
                ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i % len(colors)]], label=f'Topic {topic}', alpha=0.6, s=20)
        
        ax1.set_title('LDA Optimal\n(по композитным метрикам)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.grid(True, alpha=0.3)
        
        # BERTopic
        unique_topics_bert = list(set(bertopic_topics))
        for i, topic in enumerate(unique_topics_bert):
            if topic != -1:
                mask = np.array(bertopic_topics) == topic
                ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i % len(colors)]], label=f'Topic {topic}', alpha=0.6, s=20)
            else:  # outliers
                mask = np.array(bertopic_topics) == topic
                ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c='gray', label='Outliers', alpha=0.3, s=10)
        
        ax2.set_title('BERTopic Optimal\n(по композитным метрикам)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.grid(True, alpha=0.3)
        
        # LDA с фиксированным количеством тем
        unique_topics_lda_fixed = list(set(lda_fixed_topics))
        for i, topic in enumerate(unique_topics_lda_fixed):
            if topic != -1:
                mask = np.array(lda_fixed_topics) == topic
                ax3.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i % len(colors)]], label=f'Topic {topic}', alpha=0.6, s=20)
        
        ax3.set_title('LDA Fixed Topics\n(как у BERTopic)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('UMAP 1')
        ax3.set_ylabel('UMAP 2')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('UMAP ВИЗУАЛИЗАЦИЯ ДОКУМЕНТОВ', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('report/figures/embeddings_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_metrics_justification_report(self, 
                                          academic_info: Dict[str, Any],
                                          save_path: str) -> None:
        """Создание отчета с академическим обоснованием метрик"""
        report_content = f"""
# АКАДЕМИЧЕСКОЕ ОБОСНОВАНИЕ ВЫБОРА МЕТРИК

## Источники
- **Основные исследования**: {academic_info['composite_metrics_source']}
- **Подход**: Композитные метрики для комплексной оценки качества тематических моделей

## Веса метрик и их обоснование

### 1. Coherence CV (вес: {academic_info['weights_rationale']['coherence_cv']})
- **Назначение**: Основная метрика семантической связности тем
- **Обоснование**: Coherence CV наиболее коррелирует с человеческой оценкой качества тем
- **Источник**: Röder et al. (2015), Newman et al. (2010)

### 2. Silhouette Score (вес: {academic_info['weights_rationale']['silhouette']})
- **Назначение**: Качество кластеризации в пространстве эмбеддингов
- **Обоснование**: Измеряет насколько хорошо документы разделены по темам
- **Источник**: Rousseeuw (1987), применение в topic modeling: Angelov (2020)

### 3. Coverage (вес: {academic_info['weights_rationale']['coverage']})
- **Назначение**: Доля документов, отнесенных к определенным темам
- **Обоснование**: Важно для практического применения - модель должна классифицировать большинство документов
- **Источник**: Grootendorst (2022) - BERTopic paper

### 4. Coherence UMass (вес: {academic_info['weights_rationale']['coherence_umass']})
- **Назначение**: Дополнительная метрика разнообразия тем
- **Обоснование**: UMass coherence измеряет внутреннюю связность, дополняя CV coherence
- **Источник**: Mimno et al. (2011)

## Дополнительное сравнение с фиксированным количеством тем

**Обоснование**: {academic_info['fixed_topics_rationale']}

Это позволяет ответить на вопрос: "Какая модель лучше при одинаковых условиях?"

## Формула композитной оценки

```
Composite Score = 0.35 × Coherence_CV + 0.25 × Silhouette + 0.25 × Coverage + 0.15 × (-Coherence_UMass)
```

**Примечание**: Coherence UMass обычно отрицательная, поэтому используется с отрицательным знаком.

## Преимущества подхода

1. **Многоаспектность**: Учитывает разные аспекты качества моделей
2. **Академическая обоснованность**: Основан на проверенных исследованиях
3. **Справедливость**: Единая система оценки для всех моделей
4. **Практичность**: Учитывает как качество тем, так и покрытие документов

---
*Отчет сгенерирован автоматически системой тематического моделирования*
        """
        
        with open(f'{save_path}/metrics_justification.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("📚 Академическое обоснование сохранено в metrics_justification.md")


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