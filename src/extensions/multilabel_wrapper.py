from asreview.models.classifiers.logistic import LogisticClassifier
from asreview.models.classifiers.rf import RandomForestClassifier
from asreview.models.classifiers.svm import SVMClassifier
from asreview.models.classifiers.nb import NaiveBayesClassifier
from sklearn.multioutput import MultiOutputClassifier
from asreview.models.classifiers.base import BaseTrainClassifier
import logging

class MultiLabelWrapper(BaseTrainClassifier):
    """Wrapper for multilabel classifiers.
    
    Uses ASReview's BaseModel as the base class for multilabel classifiers.
    
    """

    def __init__(self, classifier_type: str, n_labels: int, **kwargs):
        self.classifier_type = classifier_type
        self.n_labels = n_labels
        
        # Map classifier types to their classes
        classifiers = {
            'logistic': LogisticClassifier,
            'rf': RandomForestClassifier,
            'svm': SVMClassifier,
            'nb': NaiveBayesClassifier
        }
        
        if classifier_type not in classifiers:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
            
        # Initialize the base classifier
        base_classifier = classifiers[classifier_type](**kwargs)
        
        # Wrap it with MultiOutputClassifier
        self._model = MultiOutputClassifier(base_classifier._model)

    def fit(self, X, y):
        """Fit the multilabel classifier"""
        return self._model.fit(X, y)
        
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self._model.predict_proba(X)

