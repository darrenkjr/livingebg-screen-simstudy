import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sqlite3

def calculate_wss(df, recall_point, total_docs):
    """Calculate Work Saved over Sampling at given recall point"""
    # At recall point:
    # TN + FN = number of articles marked as irrelevant up to that point
    articles_marked_irrelevant = total_docs - recall_point
    
    # WSS = (TN + FN)/N - (1 - r)
    wss = (articles_marked_irrelevant / total_docs) - (1 - 0.95)
    return wss * 100  # Convert to percentage


def read_jsonl_timing(file_path):
    """Read JSONL timing file and convert to DataFrame."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def get_recall_data(model_id):
    """Get recall data from SQL database."""
    result_dir = Path(__file__).parent / 'results'
    eval_set_path = Path(__file__).parent / 'dataset' / 'eval_data_unique.csv'
    
    try:
        result_sql_path = result_dir / 'feature_tfidf' / 'reviews' / f'{model_id}' / 'results.sql'
        with sqlite3.connect(result_sql_path) as conn:
            df = pd.read_sql_query("SELECT * FROM results", conn)
        
        evalset = pd.read_csv(eval_set_path)
        df['cumulative_relevant'] = df['label'].cumsum()
        df['recall'] = df['cumulative_relevant'] / len(evalset)
        return df
    except Exception as e:
        print(f"Error processing {model_id}: {e}")
        return None

def find_95_recall_point(df):
    """Find the index where rolling average recall first reaches 95%."""
    if df is not None:
        rolling_recall = df['recall'].rolling(window=100).mean()
        mask = rolling_recall >= 0.95
        if mask.any():
            return mask.idxmax()  # Return the index where 95% recall is first reached
    return None

# Setup paths and read data
log_dir = Path(__file__).parent / 'logs'
retrain_path = log_dir / 'binary_logistic_None_49eafaf8_iteration_timings.jsonl'
incremental_path = log_dir / 'sgd_logistic_None_2aaf1865_iteration_timings.jsonl'

retrain_df = read_jsonl_timing(retrain_path)
incremental_df = read_jsonl_timing(incremental_path)
retrain_recall = get_recall_data('binary_logistic_None_49eafaf8')
incremental_recall = get_recall_data('sgd_logistic_None_2aaf1865')

# Calculate 95% recall points
retrain_95_point = find_95_recall_point(retrain_recall)
incremental_95_point = find_95_recall_point(incremental_recall)

## Figure 1: Training Times
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Plot only rolling averages (no raw data)
window_size = 50
retrain_rolling_time = retrain_df['train_time'].rolling(window=window_size).mean()
incremental_rolling_time = incremental_df['train_time'].rolling(window=window_size).mean()

plt.plot(retrain_df['labeled_count'], retrain_rolling_time,
         label='Retrain Model', color='blue', linewidth=2.5)
plt.plot(incremental_df['labeled_count'], incremental_rolling_time,
         label='Incremental Model', color='orange', linewidth=2.5)

# Add vertical lines for 95% recall points
if retrain_95_point is not None:
    plt.axvline(x=retrain_95_point, color='blue', linestyle=':', alpha=0.5,
                label='Retrain 95% Point')
if incremental_95_point is not None:
    plt.axvline(x=incremental_95_point, color='orange', linestyle=':', alpha=0.5,
                label='Incremental 95% Point')

plt.title('Training Time Comparison: Retrain vs Incremental Learning')
plt.xlabel('Labeled Articles')
plt.ylabel('Training Time (seconds)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_time_comparison.png')
plt.close()

# Figure 2: Recall Performance
plt.figure(figsize=(12, 6))
if retrain_recall is not None and incremental_recall is not None:
    # Plot only rolling averages (no raw data)
    window_size = 100
    retrain_rolling = retrain_recall['recall'].rolling(window=window_size).mean()
    incremental_rolling = incremental_recall['recall'].rolling(window=window_size).mean()
    
    # Only plot the rolling averages with thicker lines since they're the main focus
    plt.plot(retrain_recall.index, retrain_rolling,
             label='Retrain Model', color='blue', linewidth=2.5)
    plt.plot(incremental_recall.index, incremental_rolling,
             label='Incremental Model', color='orange', linewidth=2.5)

    # Add horizontal 95% recall target line
    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95% Recall Target')

    # Add vertical lines where 95% recall is reached
    if retrain_95_point is not None:
        plt.axvline(x=retrain_95_point, color='blue', linestyle=':', alpha=0.5, 
                    label='Retrain 95% Point')
    if incremental_95_point is not None:
        plt.axvline(x=incremental_95_point, color='orange', linestyle=':', alpha=0.5, 
                    label='Incremental 95% Point')

plt.title('Recall Performance: Retrain vs Incremental Learning')
plt.xlabel('Number of Screened Articles')
plt.ylabel('Recall')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('recall_comparison.png')
plt.close()



total_articles = len(retrain_recall)

print("\nRetrain Model:")
print(f"- 95% recall reached at: {retrain_95_point:,} articles")
wss_retrain = calculate_wss(retrain_recall, retrain_95_point, total_articles)
print(f"- WSS@95: {wss_retrain:.1f}%")

print("\nIncremental Model:")
print(f"- 95% recall reached at: {incremental_95_point:,} articles")
wss_incremental = calculate_wss(incremental_recall, incremental_95_point, total_articles)
print(f"- WSS@95: {wss_incremental:.1f}%")

print("\nTraining Time Summary Statistics:")
print("\nRetrain Model:")
print(retrain_df['train_time'].describe())
print("\nIncremental Model:")
print(incremental_df['train_time'].describe())

print("\nRecall Summary Statistics:")
print("\nRetrain Model:")
print(retrain_recall['recall'].describe())
print("\nIncremental Model:")
print(incremental_recall['recall'].describe())