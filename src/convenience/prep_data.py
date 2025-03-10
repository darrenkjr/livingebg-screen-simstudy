import pandas as pd 
import pyarrow as pa 
from pathlib import Path 

datadir = Path(__file__).parent / 'dataset' 

def label_data(search_result_path : Path, matched_results_path: Path, missed_results_path: Path): 
    search_results = pd.read_parquet(search_result_path)
    #clean id
    search_results['id'] = search_results['id'].str.replace('https://openalex.org/', '').str.lower().str.strip()
    matched_results = pd.read_parquet(matched_results_path)
    missed_results = pd.read_parquet(missed_results_path)
    eval_set = pd.concat([matched_results, missed_results])
    #create matrix structure 

    question_ids = eval_set['question_id'].unique()

    label_matrix = pd.DataFrame(
        0, 
        index = search_results.index, 
        columns = [f'label_{qid}' for qid in question_ids]
    )
        # Chain multiple IDs in priority order
    eval_set['dedup_id'] = (
        eval_set['retrieved_oa_id']
        .fillna(eval_set['retrieved_pubmed_id'] + '_pubmed')
        .fillna(eval_set['retrieved_embase_id'] + '_emb')
        .fillna(eval_set['included_article_id'] + '_source')  # Final fallback
    )
    # Create a mapping of dedup_id to list of topics first
    eval_set_dedupe_bytopic = eval_set.groupby('question_id').apply(lambda x: x.drop_duplicates('dedup_id', keep = 'first')).reset_index(drop = True)
    eval_set_dedupe = eval_set_dedupe_bytopic.drop_duplicates('dedup_id', keep = 'first')
    article_topics = {}
    for _, row in eval_set_dedupe_bytopic.iterrows():
        dedup_id = row['dedup_id']
        topic = row['question_id']
        
        if dedup_id not in article_topics:
            article_topics[dedup_id] = []
        article_topics[dedup_id].append(topic)
    eval_matrix = pd.DataFrame(
        0, 
        index = eval_set_dedupe.index, 
        columns = [f'label_{qid}' for qid in question_ids]
    )
    # Fill the matrix using the article_topics mapping
    for idx, row in eval_set_dedupe.iterrows():
        dedup_id = row['dedup_id']
        if dedup_id in article_topics:
            for topic in article_topics[dedup_id]:
                eval_matrix.loc[idx, f'label_{topic}'] = 1
    

    #fill 1s where there a matches 
    for qid in question_ids: 
        match = eval_set[eval_set['question_id'] == qid]
        label_matrix.loc[search_results['id'].isin(match['retrieved_oa_id']), f'label_{qid}'] = 1


    relevant_col = ['id', 'title', 'abstract']
    labelled_data = pd.concat([search_results[relevant_col], label_matrix], axis = 1)
    relevant_eval_col = ['dedup_id', 'included_reference', 'title', 'abstract']
    eval_data_unique = pd.concat([eval_set_dedupe[relevant_eval_col], eval_matrix], axis = 1)

    # Get all label columns (those that start with 'label_')
    label_columns = [col for col in eval_data_unique.columns if col.startswith('label_')]

    # Sum all label values for each article
    sum_labels_simdata = labelled_data[label_columns].sum(axis=1).sum()
    sum_labels_evaldata = eval_data_unique[label_columns].sum(axis=1).sum()
    print(f"Total number of relvant labels: {sum_labels_simdata}")
    print(f"Total number of relvant labels: {sum_labels_evaldata}")

    return labelled_data, eval_data_unique


if __name__ == '__main__': 
    datadir = Path(__file__).parent.parent / 'dataset' 
    search_result_path = datadir / 'oa_overarching_consolidated_boolkw_search_results.parquet'
    matched_results_path = datadir / 'matched_results_oa_overarching_boolkw_search_raw.parquet'
    missed_results_path = datadir / 'missed_results_oa_overarching_boolkw_search_raw.parquet'
    output_path = datadir / 'labelled_data.csv'
    eval_output_path = datadir / 'eval_data_unique.csv'
    labelled_data, eval_data_unique = label_data(search_result_path, matched_results_path, missed_results_path)
    labelled_data.to_csv(output_path)
    eval_data_unique.to_csv(eval_output_path)

