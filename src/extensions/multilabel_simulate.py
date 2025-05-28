from asreview.review import ReviewSimulate
import numpy as np 
import pandas as pd 
from asreview.project import open_state
from tqdm import tqdm
from asreview.review.base import LABEL_NA
from asreview.models.classifiers.sgd_wrapper import IncrementalClassifier
import timeit

class ExtendedSimulate(ReviewSimulate):
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
        sgd_flag = False,
        no_retrain_flag = False,
        adaptive_retrain_flag = False,
        logger = None
    ): 
                
        # Store multilabel info
        self.n_labels = n_labels
        self.label_matrix = label_matrix
        self.label_columns = label_columns
        self.random_state = np.random.RandomState(init_seed)
        self.eval_set = eval_set
        self.sgd_flag = sgd_flag
        self.no_retrain_flag = no_retrain_flag
        self.adaptive_retrain_flag = adaptive_retrain_flag
        self.logger = logger
        self.trained_record_ids = set()
        
        # Initialize error tracking for adaptive retraining
        if self.adaptive_retrain_flag:
            self.overconfident_window = []  # Track overconfident errors
            self.underconfident_window = [] # Track underconfident errors
            self.window_size = 4   # Number of batches to track
        
        # If we have a label matrix, use our custom prior selection
        if self.label_matrix is not None and prior_indices is None:
            self.selected_articles = set()
            self.covered_topics = set()
            prior_indices = self._prior_knowledge()
        # if self.multilabel_flag == True: 
        #     self.classifier = MultilabelClassifier(model, n_jobs=-1)
        if self.sgd_flag == True: 
            self.classifier = IncrementalClassifier(model)
        elif self.no_retrain_flag == True or self.adaptive_retrain_flag == True: 
            self.classifier = model #use th batch model (retrain on entire dataset)


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
            eval_set=eval_set,
            logger=self.logger
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
        iteration = 1 
        while not self._stop_review(iteration = iteration, labeled_count = len(self.labeled), logger = self.logger):
            start_time = timeit.default_timer()

            if self.sgd_flag == True: 
                start_time_sgd = timeit.default_timer()
                self.train()
                end_time_sgd = timeit.default_timer()
                sgd_time = end_time_sgd - start_time_sgd
                self.logger.log_iteration_timings(iteration=iteration, sgd_time=sgd_time)

            if self.no_retrain_flag == True: 
                if iteration == 1: 
                    start_time_initial = timeit.default_timer()
                    self.train()
                    end_time_initial = timeit.default_timer()
                    initial_train_time = end_time_initial - start_time_initial
                    self.logger.log_iteration_timings(iteration=iteration, initial_train_time=initial_train_time)
                else: 
                    pass

            if self.adaptive_retrain_flag == True:
                if iteration == 1: 
                    start_time_initial = timeit.default_timer()
                    self.train()
                    end_time_initial = timeit.default_timer()
                    initial_train_time = end_time_initial - start_time_initial
                    self.logger.log_iteration_timings(iteration=iteration, initial_train_time=initial_train_time)

            end_time_train = timeit.default_timer()
            train_time = end_time_train - start_time

            # Query for new records to label.
            start_time_query = timeit.default_timer()
            record_ids = self._query(self.n_instances)
            end_time_query = timeit.default_timer()
            query_time = end_time_query - start_time_query

            # Label the records.
            start_time_label = timeit.default_timer()
            labels = self._label(record_ids)
            end_time_label = timeit.default_timer()
            label_time = end_time_label - start_time_label

            #check for errros beween predictions and errors 
            if self.adaptive_retrain_flag == True: 
                start_time = timeit.default_timer()
                self._error_adaptive_retrain(record_ids, labels)
                end_time = timeit.default_timer()
                total_time = end_time - start_time
                self.logger.log_iteration_timings(iteration=iteration, adaptive_retrain_time=total_time)
            # monitor progress here
            pbar_rel.update(sum(labels))
            pbar_total.update(len(labels))
            end_time_total = timeit.default_timer() - start_time
            self.logger.log_iteration_timings(iteration=iteration, train_predict_time=train_time, query_reranking_time=query_time, label_time=label_time, total_iteration_time = end_time_total, labeled_count = len(self.labeled))

            iteration +=1 
            
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
        current_labeled_ids = set(self.labeled['record_id'])

        y_sample_input = (
            pd.DataFrame(self.record_table)
            .merge(self.labeled, how="left", on="record_id")
            .loc[:, "label"]
            .fillna(LABEL_NA) 
            .to_numpy()
        )
        train_idx = np.where(y_sample_input != LABEL_NA)[0]

        if self.sgd_flag == True: 
            #fit on new data 
            new_record_ids = current_labeled_ids - self.trained_record_ids
            if new_record_ids:
                self._incremental_fit(self.X, y_sample_input, train_idx, new_record_ids)


        else: 
            #fit on *all* data
            X_train, y_train, all_idx = self.balance_model.sample(self.X, y_sample_input, train_idx)
            #grab y_train indices for multiabel case 
            # if self.multilabel_flag == True: 
            #     y_train = self.label_matrix[all_idx]
            # Fit the classifier on the training data.
            self.classifier.fit(X_train, y_train)

        self.trained_record_ids = current_labeled_ids
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

        self.training_set = len(current_labeled_ids)


    def _incremental_fit(self, X, y, train_idx, new_record_ids):
        """Fit the classifier on latest training data."""
        # Get indices for the new records
        record_ids = pd.DataFrame(self.record_table)['record_id'].values
        new_idx = [i for i in train_idx if record_ids[i] in new_record_ids]
        new_idx = np.array(new_idx) 
        try:
            X_new, y_new, all_idx_new = self.balance_model.sample(X, y, new_idx)
        except ZeroDivisionError:
            #fall back if the new sampling dataset has either only all irrelvant or only all relevant - this is the same as th simple sampling strategy
            X_new, y_new = X[new_idx], y[new_idx]
        
        self.classifier.partial_fit(X_new, y_new)
        
        # Update the trained records after incremental training
        self.trained_record_ids.update(new_record_ids)

    def _label(self, record_ids, prior=False):
        # Call parent method to handle standard labeling
        labels = super()._label(record_ids, prior)
        
        return labels

    def _error_adaptive_retrain(self, record_ids, labels):
        """Check for prediction errors and retrain if error rate exceeds threshold.
        
        Args:
            record_ids: List of record IDs that were just labeled
            labels: List of true labels for those records
        """
        # Get the probabilities we already predicted during query
        error_threshold = 0.10
        record_indices = [np.where(self.record_table == rid)[0][0] for rid in record_ids]
        predictions = self.last_probabilities[record_indices]
        
        # Check for overconfident errors (>0.80 confidence but actually irrelevant)
        overconfident_errors = np.logical_and(predictions > 0.85, labels == 0)
        
        # Check for underconfident errors (<0.20 confidence but actually relevant)
        underconfident_errors = np.logical_and(predictions < 0.15, labels == 1)
        
        # Add errors to respective windows
        self.overconfident_window.extend(overconfident_errors)
        self.underconfident_window.extend(underconfident_errors)
        
        # Calculate error rates when we have enough iterations
        window_size_records = self.window_size * self.n_instances
        if len(self.overconfident_window) >= window_size_records or len(self.underconfident_window) >= window_size_records:
            # Calculate overconfident error rate
            overconfident_errors = sum(self.overconfident_window[-window_size_records:])
            overconfident_rate = overconfident_errors / window_size_records
            
            # Calculate underconfident error rate
            underconfident_errors = sum(self.underconfident_window[-window_size_records:])
            underconfident_rate = underconfident_errors / window_size_records
            
            # If either error rate exceeds threshold, retrain on all data
            if overconfident_rate > error_threshold or underconfident_rate > error_threshold:  # 10% error threshold
                self.logger.info(f"Pre-retrain error rates - Overconfident: {overconfident_rate:.2f}, Underconfident: {underconfident_rate:.2f}")
                self.logger.info(f"Error rate exceeds threshold ({error_threshold}), retraining model")
                self.train()  # Retrain on all labeled data
                self.overconfident_window = []  # Reset error tracking
                self.underconfident_window = []  # Reset error tracking
            else:
                # If we didn't retrain, maintain sliding window by removing oldest records
                if len(self.overconfident_window) > window_size_records:
                    self.overconfident_window = self.overconfident_window[-window_size_records:]
                    self.underconfident_window = self.underconfident_window[-window_size_records:]