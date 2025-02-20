from asreview.review import ReviewSimulate
from asreview.exceptions import *
from asreview.project import ProjectExistsError
from asreview import ASReviewData, ASReviewProject
from pathlib import Path
from extensions import * 
from convenience.logging_config import LoggerConfig
import pandas as pd
from asreview.models.classifiers import NaiveBayesClassifier, LogisticClassifier, RandomForestClassifier, SVMClassifier
from asreview.models.feature_extraction import Tfidf
from asreview.models.feature_extraction.sbert import SBERT
from asreview.models.feature_extraction.specter2 import specter2
from asreview.models.feature_extraction.modernbert import modernbert
from asreview.models.query import MaxQuery
from asreview.models.balance import DoubleBalance
from extensions.stopping_criteria import CreateSimulationStoppingCriterion   



logger = LoggerConfig.setup_logger(
    logger_name="livingebg_screen_singlelabel_workflow", 
    log_dir=Path(__file__).parent / 'logs'
)

logger.info("Starting singlelabel workflow")
logger.info("Loading labelled data")


datadir = Path(__file__).parent / 'data'
resultdir = Path(__file__).parent / 'results'
resultdir.mkdir(exist_ok=True)
#topic specific workflow 

# Create a project object and folder
project_path= resultdir / "api_simulation_singlelabel"
project_datadir = project_path / 'data'
try:
    project = ASReviewProject.create(
        project_path= project_path,
        project_id="1",
        project_mode="simulate",
        project_name="pcos_ebg_singlelabel_workflow",
    )
except ProjectExistsError:
    logger.info(f"Loading project: {project_path}")
    project = ASReviewProject(project_path)

assert project_datadir.exists(), f"Data directory does not exist: {project_datadir}"
try:
    assert len(list(project_datadir.glob('*.csv'))) == 38, f"Data directory does not have 38 files: {project_datadir}"

except AssertionError:
    logger.warning(f"Data directory does not have 40 files: {project_datadir}")
    logger.info("Populating data directory with data from labelled_data.csv")
    labelleddata_dir = Path(__file__).parent / 'dataset' / 'labelled_data.csv'
    labelled_data = pd.read_csv(labelleddata_dir)
    label_cols = [col for col in labelled_data.columns if 'label' in col]


    base_cols = ['id', 'title', 'abstract']

    for col in label_cols: 
        topic_label_df = labelled_data[base_cols + [col]].copy()
        topic_label_df.rename(columns={col: 'label'}, inplace=True)
        qid = col.partition('label_')[2]
        clean_id = qid.replace('/', '-').replace('.', '-')
        logger.info(f"Prepping labelled data for question id: {qid}")
        topic_label_df.to_csv(project_datadir / f'pcosebg_{clean_id}_labelled.csv', index=False)


classifier_interest = ['lr', 'rf', 'svm', 'nb']
feature_extract_interest = ['modernbert','specter2','sbert', 'tfidf']
stopcriterion_interest = ['statistical','consecutive_irrelevant','time']

classifer_dct = {
    'lr': LogisticClassifier(),
    'rf': RandomForestClassifier(),
    'svm': SVMClassifier(),
    'nb': NaiveBayesClassifier()
}
feature_extract_dct = {
    'tfidf': Tfidf(),
    'specter2': specter2(),
    'sbert': SBERT(),
    'modernbert': modernbert()
}
#use default query strategies 
query_model = MaxQuery()
balance_model = DoubleBalance()

for file in project_datadir.glob('*.csv'):

    logger.info(f"Adding dataset: {file.name}")
    project.add_dataset(file.name)
    data_obj = ASReviewData.from_file(file)

    for feature_extract in feature_extract_interest:
        for classifier in classifier_interest:
            for stopcriterion in stopcriterion_interest:
                logger.info(f"Running singlelabel workflow with classifier: {classifier}, feature_extract: {feature_extract}, stopcriterion: {stopcriterion}")
                train_model = classifer_dct[classifier]
                feature_model = feature_extract_dct[feature_extract]
                for stopping_criterion, params in CreateSimulationStoppingCriterion(stopcriterion):
                    if stopcriterion == 'statistical':
                        logger.info(f"Running singlelabel workflow with classifier: {classifier}, feature_extract: {feature_extract}, stopcriterion: {stopcriterion}, recall target: {params[0]}, pval target: {params[1]}")
                    else: 
                        logger.info(f"Running singlelabel workflow with classifier: {classifier}, feature_extract: {feature_extract}, stopcriterion: {stopcriterion}, params: {params}")
                
                    #create simulation object 
                    logger.info(f"Creating simulation object")
                    reviewer_sim = ReviewSimulate(
                        as_data = data_obj,
                        model = train_model, 
                        query_model = query_model,
                        balance_model = balance_model,
                        feature_model = feature_model,
                        n_prior_included = 1, 
                        n_prior_excluded = 1, 
                        n_instances = 10, 
                        project = project, 
                        stop_if = stopping_criterion, 
                        write_interval = 10 # match with n_instances
                    )
                    logger.info(f"Reviewing dataset")
                    project.update_review(status = "review")
                    try: 
                        reviewer_sim.review()
                        project.mark_review_finished()
                        logger.info(f"Simulation finished for params: classifier: {classifier}, feature_extractor: {feature_extract}, stopcriterion: {stopcriterion} params: {params}")
                        
                    except Exception as e:
                       
                        raise 

project.export(resultdir / f"{project.project_name}.asreview")


















