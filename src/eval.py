import json
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns 
from pathlib import Path
import json 
import ast 
import re
from dataclasses import dataclass, asdict
import uuid 

@dataclass
class metrics:
    recall: float
    precision: float
    f1score: float
    f2score: float
    f3score: float
    adjusted_recall: float
    adjusted_precision: float
    adjusted_f1score: float
    adjusted_f2score: float
    adjusted_f3score: float

class eval:

    def __init__(self): 
        self.result_dir = Path(__file__).parent / 'results'
        self.eval_set_path = Path(__file__).parent / 'dataset' / 'eval_data_unique.csv'


    def prep_result_metadata(self): 

        '''
        Prepares result metadata dataframe for evaluation after raw simulation runs. 

        Returns: 
            result metadata dataframe 
        '''

        _dct = {} 
        for file in self.result_dir.glob('*.json'):
            uid = json.load(open(file))['simreview_id']
            feature = json.load(open(file))['feature_extraction']
            classifier = json.load(open(file))['classifier']
            stop_criterion = json.load(open(file))['stop_criterion']
            stop_params = json.load(open(file))['stop_params']
            if stop_criterion == 'statistical': 
                stop_params = self.parse_np_tuple(stop_params)
            result_sql_path = self.result_dir / f'feature_{feature}' / 'reviews' / f'{uid}' / 'results.sql'
            dataset_path = self.result_dir / f'feature_{feature}' / 'data' / 'dataset.csv'
            _dct[uid] = {
                'feature': feature,
                'classifier': classifier,
                'stop_criterion': stop_criterion,
                'stop_params': stop_params,
                'result_sql_path': result_sql_path,
                'dataset_path': dataset_path, 
            }

        self.result_metadata_df = pd.DataFrame.from_dict(_dct, orient='index').reset_index().rename(columns={'index': 'uid'})


    def parse_sql_results(self,sql_path : str, dataset_path : str):
        '''
        Parse sql results into dataframe, and calclate metrics for baseline benchmarks and statistical stopping criteria 

        Returns: 
            metrics object 
        '''
        conn = sqlite3.connect(sql_path)
        #grab everything from rsults 
        results = pd.read_sql_query("SELECT * FROM results", conn)
        #set up column template when simulating time and data driven cases 
        self.col_template = results.columns.tolist()
        evalset = pd.read_csv(self.eval_set_path)
        adjusted_evalset = pd.read_csv(dataset_path)
        return self.calc_metrics(evalset, adjusted_evalset, results)


    def calc_metrics(self, evalset, adjusted_evalset, results): 
        '''
        Calculate metrics for both raw cases, and also adjusting for underlying database and search strategy results 


        Returns: 
            metrics object 
        '''


        n_relevant_found = results['label'].sum()
        overall_relevant = len(evalset)

        recall = n_relevant_found / overall_relevant
        precision = n_relevant_found / len(results)
        f1score = self.calc_fscore(recall, precision, 1)
        f2score = self.calc_fscore(recall, precision, 2)
        f3score = self.calc_fscore(recall, precision, 3)

        adjusted_relevant = adjusted_evalset['label'].sum()
        adjusted_recall = n_relevant_found / adjusted_relevant
        adjusted_precision = n_relevant_found / len(results)
        adjusted_f1score = self.calc_fscore(adjusted_recall, adjusted_precision, 1)
        adjusted_f2score = self.calc_fscore(adjusted_recall, adjusted_precision, 2)
        adjusted_f3score = self.calc_fscore(adjusted_recall, adjusted_precision, 3)

        return metrics(recall, precision, f1score, f2score, f3score, adjusted_recall, adjusted_precision, adjusted_f1score, adjusted_f2score, adjusted_f3score)



    def simulate_timedriven_stop(self, sql_path : pd.DataFrame, metadata : pd.DataFrame, dataset_path : str):

        '''
        Timedriven stopping criteria simulation 
        - Window of size stop_point * len(df)
        - Calculate metrics at each stopping point (cutoff at window size)
        - Return dataframe with result metadata, and metrics at each stopping point 
        '''

        #takes in a dataset that was screened completely, and calculats metrics at each stopping point 
        stop_points_1 = np.linspace(1.0, 10, 10)
        stop_points_2 = np.linspace(10, 100, 10)
        stop_points = np.concatenate([stop_points_1, stop_points_2])

        df = pd.read_sql_query("SELECT * FROM results", sqlite3.connect(sql_path))
        evalset = pd.read_csv(self.eval_set_path)
        adjusted_evalset = pd.read_csv(dataset_path)
        #create unique id for each stopping point denoting a unique simulation / run / sysreview 
        
       
        result_list = []
        for stop_point in stop_points:
            #grab first stop_point reviews 
            result_row = metadata.copy()

            window_size = int(stop_point/100 * len(df))
            result_extract = df.head(window_size)
            
            metrics_dct = asdict(self.calc_metrics(evalset, adjusted_evalset, result_extract))
            result_row['stop_criterion'] = 'timedriven'
            result_row['stop_params'] = stop_point
            result_row['uid'] = f'{metadata["classifier"]}_timedriven_{uuid.uuid4().hex[:8]}'
            for key, value in metrics_dct.items():
                result_row[key] = value
            result_list.append(result_row)
            
        result_df = pd.concat(result_list, ignore_index=True)
        return result_df



    def simulate_data_driven_stop(self, sql_path : pd.DataFrame, metadata, dataset_path : str):

        '''
        Consecutive irrelevant stopping criteria simulation 
        - Sliding window of size stop_point * len(df)
        - If all reviews in the window are irrelevant, stop 
        - Calculate metrics at each stopping point 
        - Return dataframe with result metadata, and metrics at each stopping point 
        '''

        #consecutive irrelevant simulation 

        df = pd.read_sql_query("SELECT * FROM results", sqlite3.connect(sql_path))
        evalset = pd.read_csv(self.eval_set_path)
        adjusted_evalset = pd.read_csv(dataset_path)

        stop_points_1 = np.linspace(0.1, 1.0, 10)
        stop_points_2 = np.linspace(1.0, 10, 10)
        stop_points = np.concatenate([stop_points_1, stop_points_2])

        result_list = [] 
        for stop_point in stop_points:
            sliding_window = int(stop_point/100 * len(df)) 
            #loop through df in windows of size sliding_window 
            stopping_index = None 
            result_row = metadata.copy()
            for i in range(len(df) - sliding_window + 1):
                window = df['label'].iloc[i:i+sliding_window]
                if window.sum() == 0:
                    stopping_index = i + sliding_window 
                    break 
            
            result_extract = df.iloc[:stopping_index]
            metrics_dct = asdict(self.calc_metrics(evalset, adjusted_evalset, result_extract))
            result_row['stop_criterion'] = 'datadriven'
            result_row['stop_params'] = stop_point
            result_row['uid'] = f'{metadata["classifier"]}_datadriven_{uuid.uuid4().hex[:8]}'
            for key, value in metrics_dct.items():
                result_row[key] = value
            result_list.append(result_row)

        result_df = pd.concat(result_list, ignore_index=True)
        return result_df

    def run_eval(self): 
        for idx, row in self.result_metadata_df.iterrows():
            print(f'Running eval for {row["uid"]}')
            metrics_dct = asdict(self.parse_sql_results(row['result_sql_path'], row['dataset_path']))
            self.result_metadata_df.loc[idx, metrics_dct.keys()] = pd.Series(metrics_dct)
        
        #run time driven eval 
        baseline_result_metadata = self.result_metadata_df[pd.isna(self.result_metadata_df['stop_criterion'])]
        print('Running time and data driven simulations and eval')
        for idx, row in baseline_result_metadata.iterrows():
            metadata = baseline_result_metadata[['feature', 'classifier', 'result_sql_path', 'dataset_path']]
            result_metadata_timedriven = self.simulate_timedriven_stop(row['result_sql_path'], metadata, row['dataset_path'])
            result_metadata_datadriven = self.simulate_data_driven_stop(row['result_sql_path'], metadata, row['dataset_path'])
            self.result_metadata_df = pd.concat([self.result_metadata_df, result_metadata_timedriven, result_metadata_datadriven])
        
        print('Eval complete')
        return self.result_metadata_df

    def export_results(self):
        self.result_metadata_df.to_csv(self.result_dir / 'result_metadata_df.csv', index=False)


    @staticmethod
    def parse_np_tuple(str): 
        #replace numpy pattern 
        modified_string = re.sub(r'np\.int64\((\d+)\)', r'\1', str)
        #parse as tuple 
        tuple_values = ast.literal_eval(modified_string)
        values_list = list(tuple_values)
        values_list[-1] = np.int64(values_list[-1])
        return tuple(values_list)
    
    @staticmethod
    def calc_fscore(recall, precision, beta):
        return (1 + beta**2) * (recall * precision) / ((beta**2 * precision) + recall)


eval_cls = eval()
eval_cls.prep_result_metadata()
eval_cls.run_eval()
eval_cls.export_results() 