from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from asreview.models.classifiers.base import BaseTrainClassifier
import numpy as np
from sklearn.linear_model import SGDClassifier


class IncrementalClassifier(BaseTrainClassifier):
    """Multilabel classifier that maintains ASReview integration."""
    
    def __init__(self, base_model):
        # Store the original ASReview model for its properties
        self.sgd_dict = {
            'logistic' : 'log_loss', 
            'svm' : 'hinge'
        }
        self.base_asreview_model = base_model
        
        # Copy key attributes
        self.name = f"sgd_{base_model.name}"
        self.label = f"sgd_{base_model.label}"
        
        # Create multilabel classifier with the inner sklearn model
        self.sgd_model = SGDClassifier(
            loss=self.sgd_dict[base_model.name],
            warm_start=True, 
        )

        self.is_calibrated = False

        self._model = CalibratedClassifierCV(
            FrozenEstimator(self.sgd_model),
            method = 'sigmoid'
        )
        


        
    def fit(self, X, y):
        """Fit the sgd wrapped model."""
        self.sgd_model.fit(X, y)
        self._model.fit(X,y)
        self.is_calibrated = True
        return self
    
    def partial_fit(self, X, y): 
        """Partial fit the sgd wrapped model."""
        classes = np.array([0,1])
        self.sgd_model.partial_fit(X, y, classes=classes)
        if len(np.unique(y))>1: 
            self._model.fit(X, y)
            self.is_calibrated = True
        else: 
            self.is_calibrated = False 

        return self
    
    def predict_proba(self, X):
        """Get calibrated probabilities when possible, fallback to sigmoid otherwise."""
        if self.is_calibrated:
            # Use the calibrated model for predictions if possible 
            return self._model.predict_proba(X)
        else:
            scores = self.sgd_model.decision_function(X)
            proba = np.zeros_like(scores)
            mask = scores >= 0
            proba[mask] = 1.0 / (1.0 + np.exp(-scores[mask]))
            exp_scores = np.exp(scores[~mask])
            proba[~mask] = exp_scores / (1.0 + exp_scores)
            
            return np.column_stack((1-proba, proba))


    
    def full_hyper_space(self):
        """Maintain hyperparameter optimization compatibility."""
        return self.base_asreview_model.full_hyper_space()
    
    @property
    def estimators_(self):
        """Provide access to the underlying estimators in the MultiOutputClassifier."""
        return self._model.estimators_