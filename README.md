# Topic Modeling Research: LDA vs BERTopic

Comparative analysis of topic modeling approaches on Russian review corpus.

## Overview

This repository contains a comprehensive comparison study of two topic modeling approaches:
- **Latent Dirichlet Allocation (LDA)** - classical probabilistic method
- **BERTopic** - modern transformer-based approach

The study is conducted on a corpus of 5,858 Russian reviews from 2GIS and Banki.ru platforms.

## Key Findings

| Metric | LDA | BERTopic | Winner |
|--------|-----|----------|--------|
| Data Coverage | 100% | 55.4% | LDA |
| Coherence C_V | 0.541 | 0.000 | LDA |
| Topic Diversity | 0.740 | 0.675 | LDA |
| Number of Topics | 5 | 68 | Depends on use case |

**Conclusion**: LDA demonstrates superior performance for practical review analysis tasks.

## Repository Structure

```
├── data/
│   ├── processed/           # Preprocessed data files
│   └── data_for_analysis.csv # Raw dataset
├── models/
│   ├── lda_model.pkl       # Trained LDA model
│   ├── bertopic_model.pkl  # Trained BERTopic model
│   ├── lda_tuning_results.csv
│   └── bertopic_tuning_results.csv
├── notebooks/
│   ├── eda_and_preprocessing.ipynb
│   ├── lda_modeling.ipynb
│   └── model_comparison.ipynb
├── report/
│   ├── research_report.md   # Full academic report
│   ├── executive_summary.md # Business summary
│   ├── model_comparison.csv # Quantitative results
│   └── figures/            # Visualizations
├── src/
│   ├── data_preprocessing.py
│   ├── lda_model.py
│   ├── bertopic_model.py
│   ├── evaluation.py
│   └── visualization.py
├── main.py                 # Main pipeline script
└── requirements.txt        # Dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- 4GB+ RAM (for BERTopic embeddings)

### Setup
```bash
git clone <repository-url>
cd itmo_ods_nlp_maluh
pip install -r requirements.txt
```

### NLTK Data
The pipeline automatically downloads required NLTK resources on first run.

## Usage

### Quick Start
Run the complete analysis pipeline:
```bash
python main.py
```

### Command Line Options
```bash
python main.py --data data/data_for_analysis.csv --output-dir results/
```

Available arguments:
- `--data`: Path to input CSV file (default: data_for_analysis.csv)
- `--output-dir`: Output directory for results (default: current directory)
- `--lda-topics`: Range of topics for LDA tuning (default: 5,10,15,20,25,30)
- `--bertopic-clusters`: Min cluster sizes for BERTopic (default: 10,15,20,30)

### Pipeline Steps
1. **Data Preprocessing**: Text cleaning, tokenization, lemmatization
2. **LDA Modeling**: Parameter tuning and training
3. **BERTopic Modeling**: Embedding creation and clustering
4. **Evaluation**: Multiple metrics comparison
5. **Visualization**: Results plotting and analysis

## Data Format

Input CSV must contain columns:
- `text`: Review text content
- `источник`: Source platform (optional for grouping)

## Methodology

### Preprocessing
- HTML/URL removal
- Russian text tokenization (NLTK)
- Lemmatization (pymorphy2)
- Stopword filtering
- Document length filtering (>10 characters)

### Models Configuration

#### LDA
- Library: Gensim LdaMulticore
- Optimization metric: Coherence C_V
- Parameters: alpha='symmetric', eta='auto', passes=10
- Topic range: 5-30

#### BERTopic
- Embeddings: paraphrase-multilingual-MiniLM-L12-v2
- Dimensionality reduction: UMAP (n_components=5)
- Clustering: HDBSCAN (min_cluster_size tuning)
- Vectorization: CountVectorizer with n-grams

### Evaluation Metrics
- **Coherence C_V**: Semantic consistency of topics
- **Coherence U_Mass**: Statistical topic quality
- **Silhouette Score**: Clustering quality in embedding space
- **Topic Diversity**: Proportion of unique words across topics
- **Coverage**: Percentage of documents assigned to topics
- **Perplexity**: Model likelihood (LDA only)

## Results

### LDA Topics (5 identified)
1. Banking documents and transactions
2. Banking cards and customer service
3. Account operations and money transfers
4. Digital payments and services
5. Service quality in branches

### BERTopic Topics (68 identified, top 5)
1. Gratitude expressions (235 documents)
2. Mobile application issues (208 documents)
3. Mortgage and real estate (171 documents)
4. General banking questions (145 documents)
5. Working hours (141 documents)

## Technical Requirements

### Dependencies
- pandas>=1.5.0
- numpy>=1.21.0
- scikit-learn>=1.1.0
- gensim>=4.2.0
- bertopic>=0.15.0
- sentence-transformers>=2.2.0
- pymorphy2>=0.9.0
- nltk>=3.7.0

### Hardware Recommendations
- **Minimum**: 8GB RAM, 2 CPU cores
- **Recommended**: 16GB RAM, 4+ CPU cores
- **Storage**: 2GB free space for models

## API Reference

### Core Classes

#### TextPreprocessor
```python
from src.data_preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
processed_texts = preprocessor.preprocess_corpus(texts)
```

#### LDAModel
```python
from src.lda_model import LDAModel

lda = LDAModel()
lda.prepare_corpus(processed_texts)
lda.train(num_topics=5)
```

#### BERTopicModel
```python
from src.bertopic_model import BERTopicModel

bertopic = BERTopicModel()
bertopic.train(documents=texts, min_cluster_size=10)
```

## Performance Benchmarks

Runtime on Intel i7-8700K, 16GB RAM:
- Data preprocessing: ~2 minutes
- LDA parameter tuning: ~15 minutes
- BERTopic parameter tuning: ~45 minutes
- Evaluation and visualization: ~5 minutes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{topic_modeling_comparison_2025,
  title={Comparative Analysis of Topic Modeling Methods: LDA vs BERTopic on Russian Review Corpus},
  author={ITMO University NLP Research},
  year={2025},
  howpublished={\url{https://github.com/username/itmo_ods_nlp_maluh}}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions and issues:
- Check existing GitHub issues
- Create a new issue with detailed description
- Include system information and error logs

## Acknowledgments

- ITMO University Open Data Science Program
- Gensim and BERTopic development teams
- 2GIS and Banki.ru for data sources 