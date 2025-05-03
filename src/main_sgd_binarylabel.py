from extensions.multilabel_simulate import ExtendedSimulate
from asreview.exceptions import *
from asreview.project import ProjectExistsError, open_state
from asreview.state.sqlstate import SQLiteState
from asreview import ASReviewData, ASReviewProject
from pathlib import Path
from extensions import * 
from convenience.logging_config import LoggerConfig
import pandas as pd
import numpy as np
from asreview.models.classifiers import NaiveBayesClassifier, LogisticClassifier, RandomForestClassifier, SVMClassifier
from asreview.models.feature_extraction import Tfidf
from asreview.models.feature_extraction.sbert import SBERT
from asreview.models.feature_extraction.specter2 import specter2
from asreview.models.feature_extraction.biolinkbert import BioLinkBert
from asreview.models.feature_extraction.doc2vec import Doc2Vec
from asreview.models.query import MaxQuery
from asreview.models.balance import DoubleBalance
from extensions.stopping_criteria import CreateSimulationStoppingCriterion   
import uuid 
import json
import timeit


print("Starting multilabel workflow")
print("Loading labelled data")

# Setup paths
datadir = Path(__file__).parent / 'data'
resultdir = Path(__file__).parent / 'results'
resultdir.mkdir(exist_ok=True)



# Load the multilabel dataset
labelleddata_dir = Path(__file__).parent / 'dataset' / 'labelled_data.csv'
labelled_data = pd.read_csv(labelleddata_dir)
evaldata_dir = Path(__file__).parent / 'dataset' / 'eval_data_unique.csv'
eval_data_unique = pd.read_csv(evaldata_dir)
label_cols = [col for col in labelled_data.columns if 'label' in col]
n_labels = len(label_cols)

# Create label matrix
label_matrix = labelled_data[label_cols].values

# Create dataset with combined label (relevant if relevant to ANY topic)
multilabel_df = labelled_data.copy()
multilabel_df['label'] = (multilabel_df[label_cols].sum(axis=1) > 0).astype(int)

# Save the dataset to the project directory
dataset_path = Path(__file__).parent /'dataset' / "multilabel_dataset.csv"
multilabel_df.to_csv(dataset_path, index=False)


# Define models to test 

 #took out time and consecutive irrelevant can look at this retrospectively

classifier_dct = {
    'sgd_incremental_svm': SVMClassifier(),
    'sgd_incremental_logistic': LogisticClassifier(),
    'pool_svm': SVMClassifier(),
    'pool_logistic': LogisticClassifier(),
}

feature_extract_dct = {
    'tfidf': Tfidf(),
    'doc2vec': Doc2Vec(),
    'sbert': SBERT(),
    'specter2': specter2(),
    'biolinkbert': BioLinkBert()
}

# Use default query and balance strategies
query_model = MaxQuery()
balance_model = DoubleBalance()


# Store simulation metadata
simconfig_list = []
stopcriterion_interest = ['statistical','time', 'consecutive_irrelevant']
for feature_extract in feature_extract_dct.keys():
    print(f"Starting simulations with feature extraction: {feature_extract}")
    
    # Create one project per feature extraction method
    feature_project_path = resultdir / f"feature_{feature_extract}"

    
    # Create new project
    try: 
        feature_project = ASReviewProject.create(
            project_path=feature_project_path,
            project_id=f"feature_{feature_extract}",
            project_mode="simulate",
            project_name=f"Feature Extraction: {feature_extract}"
        )
    except ProjectExistsError: 
        print(f"Project for feature extraction: {feature_extract} already exists, loading existing project")
        feature_project = ASReviewProject(feature_project_path)
    
    # Setup dataset
    feature_project_datadir = feature_project_path / 'data'
    feature_project_datadir.mkdir(exist_ok=True)
    dataset_path = feature_project_datadir / "dataset.csv"
    multilabel_df.to_csv(dataset_path, index=False)
    feature_project.update_config(dataset_path="dataset.csv")
    
    # Load data for this project
    data_obj = ASReviewData.from_file(dataset_path)
    
    # Get feature extraction model
    feature_model = feature_extract_dct[feature_extract]
    
    # Now run all classifiers with this feature extraction
    for classifier in classifier_dct.keys():
        for stopcriterion in stopcriterion_interest:
            print(f"Running simulation with classifier: {classifier}, feature_extract: {feature_extract}, stopcriterion: {stopcriterion}")
            if classifier.startswith('sgd'): 
                sgd_flag = True
            elif classifier.startswith('pool'): 
                sgd_flag = False
            # Get models
            train_model = classifier_dct[classifier]
            
            # Get stopping criterion
            if stopcriterion == 'fixedrecall_benchmark': 
                stopcriterion_generator = CreateSimulationStoppingCriterion(
                    stopcriterion, 
                    label_matrix=multilabel_df, 
                    recall_target=0.95,
                    eval_set=eval_data_unique,
                )
            else: 
                stopcriterion_generator = CreateSimulationStoppingCriterion(stopcriterion)
            
            # Process all stopping criterion configurations
            for stopping_criterion, params in stopcriterion_generator:
                # Create a unique review ID for this configuration
                sim_name = f"{classifier}_{stopcriterion}"
                simreview_id = f"{sim_name}_{uuid.uuid4().hex[:8]}"  # Add short UUID for uniqueness
                timing_log_paths = Path(__file__).parent / 'results' / f'feature_{feature_extract}' / 'reviews' 
                timing_metric_logger = LoggerConfig(simreview_id).setup_logger(logger_name=f"timing_metrics_{simreview_id}", log_dir=Path(__file__).parent / timing_log_paths)


                # # Create review in this feature project
                state = SQLiteState(read_only=False)
                state._create_new_state_file(feature_project.project_path, simreview_id)
                feature_project.add_review(review_id=simreview_id)

                #initalize state 
                timing_metric_logger.info(f"Creating simulation with review ID: {simreview_id}")
                if stopping_criterion == None: 
                    write_check_interval = 1000
                else: 
                    write_check_interval = 20
                # Create simulation
                try:
                    reviewer_sim =  ExtendedSimulate(
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
                        n_instances=write_check_interval, #number of instances to label per iteration
                        project=feature_project, 
                        stop_if=stopping_criterion, 
                        write_interval=write_check_interval, #number of instances to write and check per iteration 
                        eval_total_relevant=len(eval_data_unique),
                        eval_set=eval_data_unique,
                        review_id=simreview_id, 
                        logger=timing_metric_logger, 
                        sgd_flag = sgd_flag
                    )

                    
                    # Run simulation
                    timing_metric_logger.info(f"Running simulation with review ID: {simreview_id}")
                    start_time = timeit.default_timer()
                    reviewer_sim.review() #expted length of total_eval is 764
                    end_time = timeit.default_timer()
                    sim_time = end_time - start_time
                    timing_metric_logger.info(f"Simulation finished in {end_time - start_time} seconds")
                    timing_metric_logger.log_iteration_timings(iteration=-1, sim_time=sim_time)
                   
                    
                    # Mark as finished
                    feature_project.mark_review_finished(review_id=simreview_id)
                    
                    # Store metadata
                    simulation_metadata = {
                        'simreview_id': simreview_id,
                        'feature_extraction': feature_extract, 
                        'classifier': classifier, 
                        'stop_criterion': stopcriterion, 
                        'stop_params': str(params), 
                        'simulation_time (s)': end_time - start_time
                    }


                    with open(resultdir / 'results' / f'feature_{feature_extract}'/f"simulation_metadata_{simreview_id}.json", 'w') as f:
                        json.dump(simulation_metadata, f, indent=2)
                    print(f'Simulation finisehd for id: {simreview_id}, recall: {reviewer_sim.global_recall}, classifier: {classifier}, feature_extract: {feature_extract}, stopcriterion: {stopcriterion}, params: {params}')
                    
                except Exception as e:
                    raise e

    # Export this feature project when done
    feature_project.export(resultdir / f"feature_{feature_extract}.asreview")
