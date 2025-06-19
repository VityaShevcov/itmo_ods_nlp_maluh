# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-06-19

### Added
- Initial implementation of comparative topic modeling analysis
- LDA model implementation with parameter tuning
- BERTopic model implementation with clustering optimization
- Comprehensive evaluation metrics (Coherence C_V, U_Mass, Silhouette, Topic Diversity)
- Russian text preprocessing pipeline with pymorphy2 lemmatization
- Automated pipeline execution with main.py
- Jupyter notebooks for interactive analysis
- Complete documentation and research reports
- Data visualization components

### Features
- Support for Russian language text processing
- Automated hyperparameter tuning for both models
- Multiple evaluation metrics for model comparison
- Export capabilities for trained models
- CSV output for quantitative results
- Professional reporting templates

### Models
- LDA implementation using Gensim LdaMulticore
- BERTopic implementation with multilingual sentence transformers
- UMAP dimensionality reduction
- HDBSCAN clustering
- c-TF-IDF topic representation

### Documentation
- Complete README with installation and usage instructions
- Academic research report with methodology and results
- Executive summary for business stakeholders
- API documentation for core classes

### Infrastructure
- Professional repository structure
- MIT License
- Python package setup configuration
- Comprehensive .gitignore for ML projects
- Requirements specification 