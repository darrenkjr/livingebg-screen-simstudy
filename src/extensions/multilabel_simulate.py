from asreview.review import ReviewSimulate
import numpy as np 
import pandas as pd 

class MultiLabelSimulate(ReviewSimulate):
    '''
    Wrapper for the ReviewSimulate class to simulate a multi-label classification task.
    
    '''

    def __init__(self, stop_criteria: dict, n_labels : int,  **kwargs): 
        super().__init__() # call the constructor of the parent class
        self.stop_criteria = stop_criteria
        self.n_labels = n_labels

        #implement stopping review function 

    def _stop_review(self): 
        '''
        Check if the stopping criteria are met.
        '''
        if self.pool.empty: 
            return True 
        
        
