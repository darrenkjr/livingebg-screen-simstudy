from asreview.review import ReviewSimulate
import numpy as np 
import pandas as pd 
from asreview.project import open_state
from tqdm import tqdm
from asreview.review.base import LABEL_NA
from asreview.models.classifiers.multilabel_adapter import MultilabelClassifier
from asreview.models.classifiers.sgd_wrapper import IncrementalClassifier
import timeit

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
        eval_set=None, 
        multilabel_flag = False
    ): 
                
        # Store multilabel info
        self.n_labels = n_labels
        self.label_matrix = label_matrix
        self.label_columns = label_columns
        self.random_state = np.random.RandomState(init_seed)
        self.eval_set = eval_set
        self.multilabel_flag = multilabel_flag
                # If we have a label matrix, use our custom prior selection
        if self.label_matrix is not None and prior_indices is None:
            self.selected_articles = set()
            self.covered_topics = set()
            prior_indices = self._prior_knowledge()
        if self.multilabel_flag == True: 
            self.classifier = MultilabelClassifier(model, n_jobs=-1)
        else: 
            self.classifier = IncrementalClassifier(model)

        super().__init__(
            as_data=as_data,
            model=self.classifier,
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

        self.selected_articles = set()
        self.covered_topics = set()
        prior_included = self._select_prior_included()
        prior_excluded = self._select_prior_excluded()
        return np.concatenate((prior_included, prior_excluded))
        
            #select irrelevant prior knowldege 
    
    def review(self):
        """Override to handle multilabel data properly."""
        with open_state(self.project, review_id=self.review_id, read_only=False) as s:
            pending = s.get_pending()
            if not pending.empty:
                self._label(pending)

            labels_prior = s.get_labels()

        # progress bars
        pbar_rel = tqdm(
            initial=sum(labels_prior),
            total=len(self.eval_set) if self.eval_set is not None else 764,
            desc="Relevant records found",
        )
        pbar_total = tqdm(
            initial=len(labels_prior),
            total=len(self.as_data),
            desc="Records labeled ",
        )

       # While the stopping condition has not been met:
       
        while not self._stop_review():
            # Train a new model.
            start_time = timeit.default_timer()
            self.train()
            end_time_train = timeit.default_timer()
            train_time = end_time_train - start_time
            print(f"Training time: {train_time} seconds")

            # Query for new records to label.
            record_ids = self._query(self.n_instances)

            # Label the records.
            labels = self._label(record_ids)

            # monitor progress here
            pbar_rel.update(sum(labels))
            pbar_total.update(len(labels))

        else:
            # write to state when stopped
            pbar_rel.close()
            pbar_total.close()

                
    def _select_prior_included(self): 

        prior_included = []
        
        for topic_idx in range(self.n_labels):
            # Get indices of records relevant to this topic
            topic_relevant = np.where(self.label_matrix[:, topic_idx] == 1)[0]
            
            #select relevant articls first 
            if len(topic_relevant) > 0:
                available = [_ for _ in topic_relevant if _ not in self.selected_articles]
                if available:  # Check if available is not empty
                    chosen_included = self.random_state.choice(available)
                    prior_included.append(chosen_included)
                    self.selected_articles.add(chosen_included)
                    self.covered_topics.add(topic_idx)
                else:
                    print(f"Warning: All relevant records for topic {topic_idx} have already been selected")
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
                if available:  # Check if available is not empty
                    chosen_excluded = self.random_state.choice(available)
                    prior_excluded.append(chosen_excluded)
                    self.selected_articles.add(chosen_excluded)
                else:
                    print(f"Warning: All irrelevant records for topic {topic_idx} have already been selected")
                    self.covered_topics.add(topic_idx)
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
        

    def train(self):
        """Train a new model on the labeled data."""
        # Check if both labels are available.
        new_training_set = len(self.labeled)

        y_sample_input = (
            pd.DataFrame(self.record_table)
            .merge(self.labeled, how="left", on="record_id")
            .loc[:, "label"]
            .fillna(LABEL_NA) 
            .to_numpy()
        )
        train_idx = np.where(y_sample_input != LABEL_NA)[0]

        if self.training_set > 0: 
            self._incremental_fit(self.X, y_sample_input, train_idx)

        else: 
            X_train, y_train, all_idx = self.balance_model.sample(self.X, y_sample_input, train_idx)

            #grab y_train indices 
            if self.multilabel_flag == True: 
                y_train = self.label_matrix[all_idx]
            # Fit the classifier on the training data.
            self.classifier.fit(X_train, y_train)

        # Use the query strategy to produce a ranking.
        ranked_record_ids, relevance_scores = self.query_strategy.query_multilabel(
            self.X, classifier=self.classifier, return_classifier_scores=True
        )

        self.last_ranking = pd.concat(
            [pd.Series(ranked_record_ids), pd.Series(range(len(ranked_record_ids)))],
            axis=1,
        )
        self.last_ranking.columns = ["record_id", "label"]
        # The scores for the included records in the second column.
        self.last_probabilities = relevance_scores[:, 1]

        self.training_set = new_training_set

    def _incremental_fit(self, X, y, train_idx): 
        """Fit the classifier on latest training data."""
        new_idx = train_idx[~np.isin(train_idx, range(self.training_set))]
        try: 
            X_new, y_new, all_idx_new = self.balance_model.sample(X, y, new_idx)
        except ZeroDivisionError: 
            X_new, y_new = X[new_idx], y[new_idx]

        self.classifier.partial_fit(X_new, y_new)




    def _label(self, record_ids, prior=False):
        # Call parent method to handle standard labeling
        labels = super()._label(record_ids, prior)
        
        return labels