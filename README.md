# Semi-Automated Screening in Living Guideline Maintenance

This repository contains code necessary to replicate experiments and analyses necessary for the paper: **Semi-Automated Screening in Living Guideline Maintenance: A Simulation Study of 90 Machine Learning Prioritised Screening System Configurations**

## Overview

This study simulates active learning-based screening systems for living evidence-based guidelines, comparing different machine learning configurations including:

- **Feature extraction methods**: TF-IDF, Doc2Vec, SBERT, SPECTER2, BioLinkBERT
- **Classification algorithms**: SVM, Logistic Regression
- **Training strategies**: No retraining, adaptive retraining, incremental SGD training
- **Stopping criteria**: Consecutive irrelevant, statistical, time-based

## Prerequisites

- **Python 3.12** (required)
- **Git** (for cloning the repository)
- **Pixi** (required for environment management)
- **CUDA-compatible GPU** (recommended for transformer models: SBERT, SPECTER2, BioLinkBERT)
  - CPU-only setup is possible but will be significantly slower

## Quick Start

### 1. Install Pixi

Install Pixi by following the official installation guide: [https://pixi.sh/install](https://pixi.sh/install)

### 2. Clone and Setup Repository

```bash
# Clone the repository
git clone <repository-url>
cd livingebg-screen-simstudy

# Install dependencies and activate environment
# For GPU acceleration (recommended):
pixi install --feature gpu

# For CPU-only setup (slower, especially for transformer models):
# pixi install --feature cpu

pixi run python --version  # Verify Python 3.12
```

### 3. Run the Simulation

```bash
# Navigate to the source directory
cd src

# Run the main simulation script
pixi run python main_sgd_binarylabel.py
```

## Project Structure

```
livingebg-screen-simstudy/
├── src/
│   ├── main_sgd_binarylabel.py          # Main simulation script
│   ├── dataset/                          # Input datasets
│   │   ├── labelled_data.csv            # Multi-label dataset
│   │   ├── eval_data_unique.csv         # Evaluation dataset
│   │   └── multilabel_dataset.csv       # Processed dataset
│   ├── extensions/                       # Custom ASReview extensions
│   │   ├── multilabel_simulate.py       # Extended simulation class
│   │   └── stopping_criteria.py         # Custom stopping criteria
│   ├── convenience/                      # Utility functions
│   │   └── logging_config.py            # Logging configuration
│   └── asreview/                        # Modified ASReview library
├── results/                              # Simulation outputs (created during run)
├── pixi.toml                            # Environment configuration
└── README.md                            # This file
```

## Running the Experiments

### Basic Run
```bash
cd src
pixi run python main_sgd_binarylabel.py
```

### Expected Output
The simulation will:
1. Load the multi-label dataset, and convert this to a single label setup
2. Run 90 different configurations (2 classifiers × 3 training approaches x 5 feature extractors × 3 stopping criteria) across specified stopping  criteria parameters 
3. Generate results in `src/results/` directory
4. Create timing logs for performance analysis

### Simulation Configurations

**Classifier and Training Approach Configurations (6):**
- `no_retrain_svm`: SVM without retraining
- `no_retrain_logistic`: Logistic regression without retraining
- `adaptive_retrain_svm`: SVM with adaptive retraining
- `adaptive_retrain_logistic`: Logistic regression with adaptive retraining
- `sgd_incremental_svm`: SVM with incremental SGD training
- `sgd_incremental_logistic`: Logistic regression with incremental SGD training

**Feature Extractors (5):**
- `tfidf`: TF-IDF vectorization
- `doc2vec`: Doc2Vec embeddings
- `sbert`: Sentence BERT embeddings
- `specter2`: SPECTER2 embeddings
- `biolinkbert`: BioLinkBERT embeddings

**Stopping Criteria (3):**
- `consecutive_irrelevant`: Stop after N consecutive irrelevant papers
- `statistical`: Statistical stopping criteria
- `time`: Time-based stopping

## Output Files

After running the simulation, you'll find:

```
src/results/
├── feature_tfidf/                       # Results for TF-IDF
├── feature_doc2vec/                     # Results for Doc2Vec
├── feature_sbert/                       # Results for SBERT
├── feature_specter2/                    # Results for SPECTER2
├── feature_biolinkbert/                 # Results for BioLinkBERT
└── simulation_metadata_*.json           # Metadata for each run
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this repository, please contact: darren.rajit1@monash.edu

## Acknowledgments

This work builds upon the ASReview framework for active learning in systematic reviews. I also use data from OpenAlex with thanks for access to their API. 