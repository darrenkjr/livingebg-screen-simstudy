from asreview.review import ReviewSimulate
from pathlib import Path
from extensions import * 

classifier_interest = ['logistic', 'rf', 'svm', 'nb']
stopcriterion_interest = ['time', 'consecutive_irrelevant', 'statistical']

datadir = Path(__file__).parent / 'data'

#topic specific workflow 










