from asreview.review import ReviewSimulate
from asreview.exceptions import *
from asreview.project import ProjectExistsError
from asreview import ASReviewData, ASReviewProject
from pathlib import Path
from extensions import * 
from convenience.logging_config import LoggerConfig
import pandas as pd


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
    assert len(list(project_datadir.glob('*.csv'))) == 40, f"Data directory does not have 40 files: {project_datadir}"
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

for file in project_datadir.glob('*.csv'):
    project.add_dataset(file)
    

classifier_interest = ['logistic', 'rf', 'svm', 'nb']
stopcriterion_interest = ['time', 'consecutive_irrelevant', 'statistical']















