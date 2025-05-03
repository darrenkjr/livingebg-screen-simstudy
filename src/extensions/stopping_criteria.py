from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Union
from buscarpy import calculate_h0
import pandas as pd 

class BaseStoppingCriterion(ABC):
    """Base class for all stopping criteria"""
    def __init__(self, name: str, **kwargs):
        self.name = name
        self._stopped = False
        self._stopped_at = None
        
    @abstractmethod
    def should_stop(self, state, **kwargs) -> bool:
        pass
    
    def __call__(self, state, **kwargs) -> bool:
        should_stop = self.should_stop(state, **kwargs)
        if should_stop and not self._stopped:
            self._stopped = True
        return should_stop

class TimeBasedCriterion(BaseStoppingCriterion):
    """Stop after screening a percentage of total abstracts"""
    def __init__(self, percentage: float):
        """
        Parameters
        ----------
        percentage : float
            Percentage of total abstracts to screen (0.1 to 1.0)
        """
        if not 0.01 <= percentage <= 1.0:
            raise ValueError("Percentage must be between 0.01 and 1.0")
        super().__init__(f"time_based_{percentage}")
        self.percentage = percentage
        self.name = f"time_based_{percentage}"
        
    def should_stop(self, state, **kwargs) -> bool:
        self.logger = kwargs.get('logger', None)
        total_papers = state.n_records
        reviewed_papers = len(state.get_labeled())
        total_reviewed = reviewed_papers / total_papers
        should_stop = reviewed_papers >= self.percentage * total_papers
        iteration = kwargs.get('iteration', None)
        labeled_count = kwargs.get('labeled_count', None)
        self.logger.log_iteration_timings(iteration=iteration, labeled_count = labeled_count, total_reviewed = total_reviewed)
        return should_stop

class ConsecutiveIrrelevantCriterion(BaseStoppingCriterion):
    """Stop after finding a percentage of consecutive irrelevant abstracts aka data driven strategy"""
    def __init__(self, percentage: float):
        """
        Parameters
        ----------
        percentage : float
            Percentage of consecutive irrelevant abstracts (0.01 to 0.1)
        """
        if not 0.001 <= percentage <= 0.1:
            raise ValueError("Percentage must be between 0.001 and 0.1")
        super().__init__(f"consecutive_{percentage}")
        self.percentage = percentage
        self.name = f"consecutive_irrelevant_{percentage}"
        self._last_checked_index = 0 
        
    def should_stop(self, state, **kwargs) -> bool:
        self.logger = kwargs.get('logger', None)
        labelled = state.get_labeled()
        total_papers = state.n_records
        window_size = int(self.percentage * total_papers)

        if window_size < 10: 
            window_size = 10 
        if len(labelled) < window_size:
            return False
            
        # Check last n papers where n is window_size
        new_labels = labelled['label'].iloc[self._last_checked_index:]
        #retrieve label columns 
        self._last_checked_index = len(labelled)

        if len(new_labels) >= window_size:
            recent_window = new_labels.tail(window_size)
            should_stop = (recent_window == 0).all()
            iteration = kwargs.get('iteration', None)
            labeled_count = kwargs.get('labeled_count', None)
            self.logger.log_iteration_timings(iteration=iteration, labeled_count = labeled_count, consecutive_irrelevant_window = len(recent_window))
            return should_stop  # True if all irrelevant



class StatisticalCriterion(BaseStoppingCriterion):
    """
    Stop when we have statistical confidence that we have achieved a certain recall %. 
    We use buscarpy for this. 

    Args: 
        recall_target : float
            The recall target to achieve (in float).
        pval_target : float
            The p-value target to achieve (in float).
    
    """

    def __init__(self, recall_target : float, pval_target : float = 0.05, bias : int = 1):
        self.recall_target = recall_target
        self.pval_target = pval_target
        super().__init__(f"statistical_{recall_target}_{pval_target}")
        self.name = f"statistical_{recall_target}_{pval_target}"
        self.bias = bias

    def should_stop(self, state, **kwargs) -> bool:
        '''
        Calcualte the pval of the current state and check if it is below target pval. If it is, stop.

        Return: 
            True if we should stop, False otherwise.
    
        '''
        self.logger = kwargs.get('logger', None)
        total_papers = state.n_records
        labelled = state.get_labeled()
        # evaluate null hypothesios and calculate pvalue, dont stop until pvalue is below target pval 
        current_pval = calculate_h0(N = total_papers, labels_ = labelled['label'], recall_target = self.recall_target, bias = self.bias)
        #log p-val 
        iteration = kwargs.get('iteration', None)
        labeled_count = kwargs.get('labeled_count', None)
        self.logger.log_iteration_timings(iteration=iteration, labeled_count = labeled_count, pval = current_pval)
        return current_pval < self.pval_target


def CreateSimulationStoppingCriterion(criterion_type: str, **kwargs): 
    '''
    Factory function to create stopping criteria for simulations.
    
    '''
    if criterion_type == "time":
        max_screened_limit = kwargs.get('max_screened_limit', 1.0)
        starting_limit = 0.01
        assert 0.01 <= max_screened_limit <= 1.0, "max_screened_limit must be between 0.1 and 1.0"
        small_steps = np.arange(starting_limit, 0.1, 0.01)  # 0.01 -> 0.09 in steps of 0.01
        large_steps = np.arange(0.1, max_screened_limit + 0.1, 0.1)  # 0.1 -> max_screened_limit in steps of 0.1
        for limit in np.concatenate([small_steps, large_steps]):
            limit = round(limit, 3)  # Round to 3 decimal places for precision
            yield TimeBasedCriterion(limit), limit
    
    
    elif criterion_type == "consecutive_irrelevant":
        max_irrelevant_limit = kwargs.get('max_irrelevant_limit', 0.1) # 0.1 is the max which is 5000 ish consecutive irrelevant abstracts
        starting_limit = 0.001  # 0.1% which is round 50 consecutive irrelevant abstracts
        assert 0.001 <= max_irrelevant_limit <= 0.1, "max_irrelevant_limit must be between 0.001 and 0.1" 
        small_steps = np.arange(starting_limit, 0.01, 0.001)  # 0.001 -> 0.009 in steps of 0.001
        large_steps = np.arange(0.01, max_irrelevant_limit + 0.01, 0.01)  # 0.01 -> max_irrelevant_limit in steps of 0.01
        for limit in np.concatenate([small_steps, large_steps]):
            limit = round(limit, 3)  # Round to 3 decimal places for precision
            yield ConsecutiveIrrelevantCriterion(limit), limit

    elif criterion_type == "statistical":
        recall_target = kwargs.get('recall_target', 0.95)
        pval_target = kwargs.get('pval_target', 0.05)
        starting_bias = 1 
        starting_bias_step = 1 
        bias_limit = 10 
        bias_steps = np.arange(starting_bias, bias_limit+1, starting_bias_step)
        for bias in bias_steps: 
            yield StatisticalCriterion(recall_target, pval_target, bias), (recall_target, pval_target, bias)
    
    else: 
        yield None, None
