from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Union
from buscarpy import calculate_h0, recall_frontier, retrospective_h0

class BaseStoppingCriterion(ABC):
    """Base class for all stopping criteria"""
    def __init__(self, name: str):
        self.name = name
        self._stopped = False
        self._stopped_at = None
        
    @abstractmethod
    def should_stop(self, state) -> bool:
        pass
    
    def __call__(self, state) -> bool:
        should_stop = self.should_stop(state)
        if should_stop and not self._stopped:
            self._stopped = True
            self._stopped_at = len(state.labeled)
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
        if not 0.1 <= percentage <= 1.0:
            raise ValueError("Percentage must be between 0.1 and 1.0")
        super().__init__(f"time_based_{percentage}")
        self.percentage = percentage
        
    def should_stop(self, state) -> bool:
        total_papers = len(state.record_table)
        reviewed_papers = len(state.labeled)
        return reviewed_papers >= self.percentage * total_papers

class ConsecutiveIrrelevantCriterion(BaseStoppingCriterion):
    """Stop after finding a percentage of consecutive irrelevant abstracts aka data driven strategy"""
    def __init__(self, percentage: float):
        """
        Parameters
        ----------
        percentage : float
            Percentage of consecutive irrelevant abstracts (0.01 to 0.1)
        """
        if not 0.01 <= percentage <= 0.1:
            raise ValueError("Percentage must be between 0.01 and 0.1")
        super().__init__(f"consecutive_{percentage}")
        self.percentage = percentage
        
    def should_stop(self, state) -> bool:
        if len(state.labeled) == 0:
            return False
        label_col = [col for col in state.record_table.columns if 'label_' in col]   
        total_papers = len(state.record_table)
        window_size = int(self.percentage * total_papers)

        if window_size < 10: 
            window_size = 10 
        if len(state.labeled) < window_size:
            return False
            
        # Check last n papers where n is window_size
        recent_labels = state.labeled[label_col].tail(window_size)
        #retrieve label columns 
        
        if len(label_col) > 1 : 
            #multilabel case 
            return not recent_labels.any().any() #will return true if there are no rlevant labels left 
        else: 
            return (recent_labels['label'] == 0).all() #returns true if labels are 0 



class MixedHeuristicCriterion(BaseStoppingCriterion):
    """Combined strategy using both time-based and consecutive irrelevant criteria"""
    def __init__(self, 
                 min_screened_percentage: float = 0.2,
                 consecutive_irrelevant_percentage: float = 0.05):
        """
        Parameters
        ----------
        min_screened_percentage : float
            Minimum percentage of total abstracts to screen (0.1 to 1.0)
        consecutive_irrelevant_percentage : float
            Percentage of consecutive irrelevant abstracts (0.01 to 0.1)
        """
        super().__init__(f"mixed_{min_screened_percentage}_{consecutive_irrelevant_percentage}")
        self.time_criterion = TimeBasedCriterion(min_screened_percentage)
        self.consecutive_criterion = ConsecutiveIrrelevantCriterion(consecutive_irrelevant_percentage)
        
    def should_stop(self, state) -> bool:
        # Must meet both conditions:
        # 1. Minimum percentage of abstracts screened
        # 2. Required consecutive irrelevant abstracts found
        return (self.time_criterion.should_stop(state) and 
                self.consecutive_criterion.should_stop(state))
    

class StatisticalCriterion(BaseStoppingCriterion):
    """
    Stop when we have statistical confidence that we have achieved a certain recall %. 
    We use buscarpy for this, which is based on the hypergeometric distribution. 

    Args: 
        recall_target : float
            The recall target to achieve (in float).
        pval_target : float
            The p-value target to achieve (in float).
    
    """

    def __init__(self, recall_target : float, pval_target : float = 0.05):
        self.recall_target = recall_target
        self.pval_target = pval_target
        super().__init__(f"statistical_{recall_target}_{pval_target}")

    def should_stop(self, state) -> bool:
        '''
        Calcualte the pval of the current state and check if it is below target pval. If it is, stop.

        Return: 
            True if we should stop, False otherwise.
    
        '''

        #check for multilabel case 
        label_col = [col for col in state.record_table.columns if 'label_' in col]
        if len(label_col) > 1 : 
            #multilabel case 
            #create a binary label column 
            state
        else: 
            #binary case 
            pass 



        # evaluate null hypothesios and calculate pvalue, dont stop until pvalue is below target pval 
        current_pval = calculate_h0(N = len(state.record_table), labels_ = state.labeled['label'], recall_target = self.recall_target)
        return current_pval < self.pval_target


def CreateSimulationStoppingCriterion(criterion_type: str, **kwargs): 
    '''
    Factory function to create stopping criteria for simulations.
    
    '''

    if criterion_type == "time":
        max_screened_limit = kwargs.get('max_screened_limit', 1.0)
        starting_limit = 0.1
        assert 0.1 <= max_screened_limit <= 1.0, "max_screened_limit must be between 0.1 and 1.0"
        current_limit = starting_limit
        while current_limit < max_screened_limit:
            yield TimeBasedCriterion(current_limit)
            current_limit += 0.1
    
    elif criterion_type == "consecutive_irrelevant":
        max_irrelevant_limit = kwargs.get('max_irrelevant_limit', 0.1)
        starting_limit = 0.01
        assert 0.01 <= max_irrelevant_limit <= 0.1, "max_irrelevant_limit must be between 0.01 and 0.1"
        current_limit = starting_limit
        while current_limit < max_irrelevant_limit:
            yield ConsecutiveIrrelevantCriterion(current_limit)
            current_limit += 0.01

    elif criterion_type == "statistical":
        recall_target = kwargs.get('recall_target', 0.95)
        pval_target = kwargs.get('pval_target', 0.05)
        yield StatisticalCriterion(recall_target, pval_target)

