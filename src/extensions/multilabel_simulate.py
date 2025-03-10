from asreview.review import ReviewSimulate
import numpy as np 
import pandas as pd 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import clone

class MultiLabelSimulate(ReviewSimulate):
    '''
    Wrapper for the ReviewSimulate class to simulate a multi-label classification 
    '''

    def __init__(
        self, 
        as_data,
        model,
        query_model,
        balance_model,
        feature_model,
        n_labels,
        label_matrix=None,
        label_columns=None,
        n_prior_included=1,
        n_prior_excluded=1,
        prior_indices=None,
        init_seed=None,
        write_interval=None,
        n_instances=1,
        stop_if=None,
        project=None,
        review_id=None,
        eval_total_relevant=None, 
        eval_set=None
    ): 
        # Initialize parent class first
        super().__init__(
            as_data=as_data,
            model=model,
            query_model=query_model,
            balance_model=balance_model,
            feature_model=feature_model,
            n_prior_included=n_prior_included,
            n_prior_excluded=n_prior_excluded,
            prior_indices=prior_indices,
            init_seed=init_seed,
            write_interval=write_interval,
            n_instances=n_instances,
            stop_if=stop_if,
            project=project, 
            review_id=review_id,
            eval_set=eval_set
        )
        
        # Store multilabel info
        self.n_labels = n_labels
        self.label_matrix = label_matrix
        self.label_columns = label_columns
        self.random_state = np.random.RandomState(init_seed)
        self.eval_set = eval_set

        
        # Create multilabel classifier by wrapping the base classifier
        self.multilabel_classifier = MultiOutputClassifier(clone(model._model))
        
        # Calculate total number of relevant records (unique across all topics)
        if label_matrix is not None:
            # A record is relevant if it's relevant to ANY topic
            self.relevant_mask = np.any(label_matrix == 1, axis=1)
            #adjust for the fact that we know the *true* recall, otherwise recall would be inflated
            if eval_total_relevant is not None:
                self.total_relevant = eval_total_relevant
            else: 
                self.total_relevant = np.sum(self.relevant_mask)
            self.found_relevant = set()  # Track found relevant record IDs
            
            # Initialize global recall
            self.global_recall = 0.0

    def _prior_knowledge(self):
        """Select one relevant article per topic"""
        if not hasattr(self, 'label_matrix') or self.label_matrix is None:
            # Fall back to parent implementation if no label matrix
            return super()._prior_knowledge()
        else: 
            self.selected_articles = set()
            self.covered_topics = set()
            prior_included = self._select_prior_included()
            prior_excluded = self._select_prior_excluded()
            return np.concatenate((prior_included, prior_excluded))
            
            #select irrelevant prior knowldege 

                
    def _select_prior_included(self): 

        prior_included = []
        
        for topic_idx in range(self.n_labels):
            # Get indices of records relevant to this topic
            topic_relevant = np.where(self.label_matrix[:, topic_idx] == 1)[0]
            
            #select relevant articls first 
            if len(topic_relevant) > 0:
                available = [_ for _ in topic_relevant if _ not in self.selected_articles]
                chosen_included = self.random_state.choice(available)
                prior_included.append(chosen_included)
                self.selected_articles.add(chosen_included)
                self.covered_topics.add(topic_idx)
            else: 
                print(f"Warning: No relevant records for topic {topic_idx}")
                self.covered_topics.add(topic_idx)

        return prior_included
    
    def _select_prior_excluded(self): 

        prior_excluded = []
        
        for topic_idx in range(self.n_labels): 
            topic_irrelevant = np.where(self.label_matrix[:, topic_idx] == 0)[0]
            if len(topic_irrelevant) > 0: 
                available = [_ for _ in topic_irrelevant if _ not in self.selected_articles]
                chosen_excluded = self.random_state.choice(available)
                prior_excluded.append(chosen_excluded)
                self.selected_articles.add(chosen_excluded)
        return prior_excluded

    
    def _get_labeled_matrices(self):
        """Get feature matrix X and label matrix y for labeled records."""
        # Get labeled record IDs and their indices
        labeled_record_ids = self.labeled["record_id"].values
        labeled_indices = [np.where(self.record_table == rid)[0][0] for rid in labeled_record_ids]
        

        #attempt to access feature cache if this is a rerun on the same feature extraction technique 
        X = self.X[labeled_indices]

        if self.label_matrix is not None:
            # Extract multilabel matrix for labeled records
            y_multilabel = self.label_matrix[labeled_indices]
            return X, y_multilabel

    
    def _train_model(self):
        """Train both the binary classifier and multilabel classifier."""
        # First train the standard binary classifier (parent implementation)
        super()._train_model()
        
        # Now train the multilabel classifier if we have multilabel data
        if hasattr(self, 'label_matrix') and self.label_matrix is not None:
            # Get feature matrix and multilabel matrix for labeled records
            X, y_multilabel = self._get_labeled_matrices()
            
            # Train multilabel classifier if we have enough labeled records
            if len(X) > 0 and np.any(y_multilabel.sum(axis=0) > 0):
                try:
                    self.multilabel_classifier.fit(X, y_multilabel)
                except Exception as e:
                    print(f"Error training multilabel classifier: {e}")
    
    def _get_proba(self, pool_indices):
        """Get relevance probabilities incorporating multilabel information."""
        # Get features for unlabeled records
        X_pool = self.X[pool_indices]
        # First get standard binary probabilities (parent implementation)
        binary_proba = self.classifier.predict_proba(X_pool)[:, 1]
        
        # If we have a trained multilabel classifier, use it to enhance probabilities
        if hasattr(self, 'multilabel_classifier') and hasattr(self.multilabel_classifier, 'estimators_'):
            try:
                # Get multilabel probabilities
                multilabel_proba = np.array([
                    est.predict_proba(X_pool)[:, 1] if est is not None else np.zeros(len(X_pool))
                    for est in self.multilabel_classifier.estimators_
                ]).T
                
                # Use maximum probability across all topics
                max_proba = np.max(multilabel_proba, axis=1)
                
                # Return the multilabel probability
                return max_proba
            except: 
                raise 

    def _label(self, record_ids, prior=False):
        # Call parent method to handle standard labeling
        labels = super()._label(record_ids, prior)
        
        return labels