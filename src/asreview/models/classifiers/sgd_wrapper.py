from sklearn.calibration import CalibratedClassifierCV
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
        self._model = SGDClassifier(
            loss=self.sgd_dict[base_model.name],
            warm_start=True, 
        )

        # self.calibrated_model = CalibratedClassifierCV(
        #     base_estimator=self._model,
        #     method='sigmoid',
        # ) #dont use this for now 

        
    def fit(self, X, y):
        """Fit the sgd wrapped model."""
        return self._model.fit(X, y)
    
    def partial_fit(self, X, y): 
        """Partial fit the sgd wrapped model."""
        classes = np.array([0,1])
        return self._model.partial_fit(X, y, classes=classes)
    
    def predict_proba(self, X):
        """Predict the probabilities of the sgd wrapped model."""
        if hasattr(self._model, 'predict_proba'):
            return self._model.predict_proba(X)
        else: 
            scores =  self._model.decision_function(X)
            #convert using sigmoid 
            proba = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack((1-proba, proba))


    
    def full_hyper_space(self):
        """Maintain hyperparameter optimization compatibility."""
        return self.base_asreview_model.full_hyper_space()
    
    @property
    def estimators_(self):
        """Provide access to the underlying estimators in the MultiOutputClassifier."""
        return self._model.estimators_