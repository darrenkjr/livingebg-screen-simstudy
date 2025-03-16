from sklearn.multioutput import MultiOutputClassifier
from asreview.models.classifiers.base import BaseTrainClassifier
import numpy as np
from sklearn.base import clone

class MultilabelClassifier(BaseTrainClassifier):
    """Multilabel classifier that maintains ASReview integration."""
    
    def __init__(self, base_model, n_jobs=-1):
        # Store the original ASReview model for its properties
        self.base_asreview_model = base_model
        
        # Copy key attributes
        self.name = f"multilabel_{base_model.name}"
        self.label = f"Multilabel {base_model.label}"
        
        # Create multilabel classifier with the inner sklearn model
        self._model = MultiOutputClassifier(
            clone(base_model._model),
            n_jobs=n_jobs
        )
    
    def fit(self, X, y):
        """Fit the multilabel model."""
        return self._model.fit(X, y)
    
    def predict_proba(self, X):
        """Get probabilities in a format compatible with query strategies."""
        # Get multilabel probabilities
        topic_probas = np.array([
            est.predict_proba(X)[:, 1] if est is not None else np.zeros(len(X))
            for est in self._model.estimators_
        ]).T
        
        # Maximum probability across topics
        max_proba = np.max(topic_probas, axis=1)
        
        # Format as binary probabilities
        return np.column_stack((1 - max_proba, max_proba))
    
    def full_hyper_space(self):
        """Maintain hyperparameter optimization compatibility."""
        return self.base_asreview_model.full_hyper_space()
    
    @property
    def estimators_(self):
        """Provide access to the underlying estimators in the MultiOutputClassifier."""
        return self._model.estimators_