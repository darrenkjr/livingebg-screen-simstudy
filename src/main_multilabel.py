from extensions.multilabel_simulate import MultiLabelSimulate
from asreview.exceptions import *
from asreview.project import ProjectExistsError
from asreview import ASReviewData, ASReviewProject
from pathlib import Path
from extensions import * 
from convenience.logging_config import LoggerConfig
import pandas as pd
import numpy as np
from asreview.models.classifiers import NaiveBayesClassifier, LogisticClassifier, RandomForestClassifier, SVMClassifier
from asreview.models.feature_extraction import Tfidf
from asreview.models.feature_extraction.sbert import SBERT
# from asreview.models.feature_extraction.specter2 import specter2
from asreview.models.feature_extraction.modernbert import modernbert
from asreview.models.query import MaxQuery
from asreview.models.balance import DoubleBalance
from extensions.stopping_criteria import CreateSimulationStoppingCriterion   

# Setup logging
logger = LoggerConfig.setup_logger(
    logger_name="livingebg_screen_multilabel_workflow", 
    log_dir=Path(__file__).parent / 'logs'
)

logger.info("Starting multilabel workflow")
logger.info("Loading labelled data")

# Setup paths
datadir = Path(__file__).parent / 'data'
resultdir = Path(__file__).parent / 'results'
resultdir.mkdir(exist_ok=True)

# Create a project object and folder
project_path = resultdir / "api_simulation_multilabel"
project_datadir = project_path / 'data'

try:
    project = ASReviewProject.create(
        project_path=project_path,
        project_id="2",
        project_mode="simulate",
        project_name="pcos_ebg_multilabel_workflow",
    )
except ProjectExistsError:
    logger.info(f"Loading project: {project_path}")
    project = ASReviewProject(project_path)

# Load the multilabel dataset
labelleddata_dir = Path(__file__).parent / 'dataset' / 'labelled_data.csv'
labelled_data = pd.read_csv(labelleddata_dir)
label_cols = [col for col in labelled_data.columns if 'label' in col]
n_labels = len(label_cols)

# Create label matrix
label_matrix = labelled_data[label_cols].values

# Create dataset with combined label (relevant if relevant to ANY topic)
multilabel_df = labelled_data.copy()
multilabel_df['label'] = (multilabel_df[label_cols].sum(axis=1) > 0).astype(int)

# Save the dataset to the project directory
dataset_path = project_datadir / "multilabel_dataset.csv"
multilabel_df.to_csv(dataset_path, index=False)

# Add dataset to project
dataset_name = "multilabel_dataset.csv"
project.add_dataset(dataset_name)
data_obj = ASReviewData.from_file(dataset_path)

# Define models to test 
classifier_interest = ['lr', 'rf', 'svm', 'nb']
feature_extract_interest = ['tfidf', 'modernbert', 'specter2', 'sbert', 'tfidf']
stopcriterion_interest = ['time', 'consecutive_irrelevant', 'statistical']

classifer_dct = {
    'lr': LogisticClassifier(),
    'rf': RandomForestClassifier(),
    'svm': SVMClassifier(),
    'nb': NaiveBayesClassifier()
}

feature_extract_dct = {
    'tfidf': Tfidf(),
    # 'specter2': specter2(),
    'sbert': SBERT(),
    'modernbert': modernbert()
}

# Use default query and balance strategies
query_model = MaxQuery()
balance_model = DoubleBalance()

# Run simulations with different configurations
for feature_extract in feature_extract_interest:
    for classifier in classifier_interest:
        for stopcriterion in stopcriterion_interest:
            logger.info(f"Running multilabel workflow with classifier: {classifier}, feature_extract: {feature_extract}, stopcriterion: {stopcriterion}")
            
            train_model = classifer_dct[classifier]
            feature_model = feature_extract_dct[feature_extract]
            
            for stopping_criterion, params in CreateSimulationStoppingCriterion(stopcriterion):
                if stopcriterion == 'statistical':
                    logger.info(f"Running multilabel workflow with classifier: {classifier}, feature_extract: {feature_extract}, stopcriterion: {stopcriterion}, recall target: {params[0]}, pval target: {params[1]}")
                else: 
                    logger.info(f"Running multilabel workflow with classifier: {classifier}, feature_extract: {feature_extract}, stopcriterion: {stopcriterion}, params: {params}")
                
                # Create multilabel simulation object
                logger.info(f"Creating multilabel simulation object")
                reviewer_sim = MultiLabelSimulate(
                    as_data=data_obj,
                    model=train_model, 
                    query_model=query_model,
                    balance_model=balance_model,
                    feature_model=feature_model,
                    n_labels=n_labels,
                    label_matrix=label_matrix,
                    label_columns=label_cols,
                    n_prior_included=1, 
                    n_prior_excluded=1, 
                    n_instances=10, 
                    project=project, 
                    stop_if=stopping_criterion, 
                    write_interval=10
                )
                
                # Run simulation
                logger.info(f"Reviewing dataset using multilabel approach")
                project.update_review(status="review")
                try: 
                    reviewer_sim.review()
                    project.mark_review_finished()
                    logger.info(f"Simulation finished with global recall: {reviewer_sim.global_recall:.4f}")
                    logger.info(f"Simulation finished for params: classifier: {classifier}, feature_extractor: {feature_extract}, stopcriterion: {stopcriterion} params: {params}")
                except Exception as e:
                    logger.error(f"Error during simulation: {e}")
                    raise 

# Export project
project.export(resultdir / f"{project.project_name}.asreview")