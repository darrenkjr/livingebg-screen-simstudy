import pandas as pd 
import pyarrow as pa 
from pathlib import Path 

datadir = Path(__file__).parent / 'dataset' 

def label_data(search_result_path : Path, matched_results_path: Path): 
    search_results = pd.read_parquet(search_result_path)
    #clean id
    search_results['id'] = search_results['id'].str.replace('https://openalex.org/', '').str.lower().str.strip()
    matched_results = pd.read_parquet(matched_results_path)

    #create matrix structure 
    question_ids = matched_results['question_id'].unique()
    label_matrix = pd.DataFrame(
        0, 
        index = search_results.index, 
        columns = [f'label_{qid}' for qid in question_ids]
    )

    #fill 1s where there a matches 
    for qid in question_ids: 
        match = matched_results[matched_results['question_id'] == qid]
        # Fixed chained assignment using loc
        label_matrix.loc[search_results['id'].isin(match['retrieved_oa_id']), f'label_{qid}'] = 1
    
    relevant_col = ['id', 'title', 'abstract']
    labelled_data = pd.concat([search_results[relevant_col], label_matrix], axis = 1)

    return labelled_data 


if __name__ == '__main__': 
    datadir = Path(__file__).parent.parent / 'dataset' 
    search_result_path = datadir / 'oa_overarching_consolidated_boolkw_search_results.parquet'
    matched_results_path = datadir / 'matched_results_oa_overarching_boolkw_search.parquet'

    output_path = datadir / 'labelled_data.csv'
    labelled_data = label_data(search_result_path, matched_results_path)
    labelled_data.to_csv(output_path)

