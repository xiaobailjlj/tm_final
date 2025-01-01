import pandas as pd
import numpy as np


def read_and_calculate_fleiss_kappa(csv_path):
    """
    Read bias annotation CSV and calculate Fleiss' Kappa.

    Parameters:
    csv_path (str): Path to CSV file with columns:
        id_event, id_article, article_bias, id_sentence, sentence_text, bias_score

    Returns:
    tuple: (kappa_score, summary_dict)
    """
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

    # Verify required columns
    required_columns = ['id_event', 'id_article', 'article_bias',
                        'id_sentence', 'sentence_text', 'bias_score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Create contingency table of ratings per item
    contingency_table = pd.crosstab(
        index=[df['id_article'], df['id_sentence']],
        columns=df['bias_score'],
        dropna=False
    ).fillna(0)

    N = len(contingency_table)  # Number of items (article-sentence pairs)
    n = contingency_table.sum(axis=1).iloc[0]  # Number of ratings per item
    k = len(contingency_table.columns)  # Number of possible ratings

    if N == 0:
        raise ValueError("No valid items found in the contingency table")

    # Calculate P (observed agreement)
    P = (contingency_table ** 2).sum(axis=1).sub(n).div(n * (n - 1))
    P_bar = P.mean()

    # Calculate Pe (expected agreement)
    Pe = ((contingency_table.sum() / (N * n)) ** 2).sum()

    # Calculate Fleiss' Kappa
    kappa = (P_bar - Pe) / (1 - Pe)

    # Create summary dictionary
    summary = {
        'number_of_items': N,
        'ratings_per_item': int(n),
        'number_of_categories': k,
        'unique_bias_scores': sorted(df['bias_score'].unique()),
        'total_annotations': len(df),
        'observed_agreement': float(P_bar),
        'expected_agreement': float(Pe)
    }

    return kappa, summary


def print_results(kappa, summary):
    """Print formatted results of the Fleiss' Kappa analysis."""
    print("\n=== Fleiss' Kappa Analysis ===")
    print(f"Fleiss' Kappa Score: {kappa:.3f}")
    print("\nSummary Statistics:")
    print(f"Number of items rated: {summary['number_of_items']}")
    print(f"Ratings per item: {summary['ratings_per_item']}")
    print(f"Number of rating categories: {summary['number_of_categories']}")
    print(f"Unique bias scores: {summary['unique_bias_scores']}")
    print(f"Total annotations: {summary['total_annotations']}")
    print(f"Observed agreement: {summary['observed_agreement']:.3f}")
    print(f"Expected agreement: {summary['expected_agreement']:.3f}")


# Example usage
if __name__ == "__main__":
    try:
        # csv_path = './news_bias_dataset/preprocessed_dataset.csv'  # 0.121
        # csv_path = './news_bias_dataset/preprocessed_dataset_topic_facebook.csv'  # -0.037
        # csv_path = './news_bias_dataset/preprocessed_dataset_topic_johnson.csv'  # 0.057
        # csv_path = './news_bias_dataset/preprocessed_dataset_topic_NFL.csv'  # 0.201
        # csv_path = './news_bias_dataset/preprocessed_dataset_topic_northkora.csv'  # -0.063
        # csv_path = './news_bias_dataset/preprocessed_dataset_binary.csv'  # 0.332
        # csv_path = './news_bias_dataset/preprocessed_dataset_binary_topic_facebook.csv'  # -0.020
        # csv_path = './news_bias_dataset/preprocessed_dataset_binary_topic_johnson.csv'  # 0.317
        # csv_path = './news_bias_dataset/preprocessed_dataset_binary_topic_NFL.csv'  # 0.499
        csv_path = './news_bias_dataset/preprocessed_dataset_binary_topic_NorthKora.csv'  # -0.081

        kappa, summary = read_and_calculate_fleiss_kappa(csv_path)
        print_results(kappa, summary)
    except Exception as e:
        print(f"Error: {e}")