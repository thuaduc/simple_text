"""Data loader for Cochrane sentence-level simplification data."""

import ast
import pandas as pd
from typing import List, Tuple


def parse_simple_column(simple_str: str) -> List[str]:
    """
    Parse the 'simple' column which is stored as string representation of list.
    
    Args:
        simple_str: String representation of list, e.g., "['sentence1', 'sentence2']"
    
    Returns:
        List of simplified sentences
    """
    if pd.isna(simple_str) or simple_str == "" or simple_str == "[]":
        return []
    
    try:
        # Use ast.literal_eval to safely parse the string representation
        result = ast.literal_eval(simple_str)
        if isinstance(result, list):
            return result
        else:
            return [str(result)]
    except (ValueError, SyntaxError):
        # If parsing fails, return as single-item list
        return [simple_str.strip()]


def load_cochrane_sentences(
    split: str = "test",
    data_dir: str = "cochrane/data",
    rephrase_only: bool = True,
) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    """
    Load Cochrane sentence-level simplification data.
    
    Args:
        split: One of 'train', 'val', or 'test'
        data_dir: Path to data directory containing CSV files
        rephrase_only: If True, load rephrase-only subset CSVs
    
    Returns:
        Tuple of (complex_sentences, simple_references, labels, pair_ids)
        - complex_sentences: List of complex input sentences
        - simple_references: List of lists (multiple references per input)
        - labels: List of simplification operation labels
        - pair_ids: List of document IDs
    """
    prefix = "cochraneauto_sents_rephrase" if rephrase_only else "cochraneauto_sents"
    csv_path = f"{data_dir}/{prefix}_{split}.csv"
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    complex_sentences = []
    simple_references = []
    labels = []
    pair_ids = []
    
    for _, row in df.iterrows():
        complex_sent = row['complex']
        simple_list = parse_simple_column(row['simple'])
        label = row['label']
        pair_id = row['pair_id']
        
        # Skip entries with empty complex sentence
        if pd.isna(complex_sent) or complex_sent.strip() == "":
            continue
        
        complex_sentences.append(complex_sent.strip())
        simple_references.append(simple_list)
        labels.append(label)
        pair_ids.append(pair_id)
    
    print(f"Loaded {len(complex_sentences)} sentence pairs from {split} set")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    return complex_sentences, simple_references, labels, pair_ids


def load_rephrase_csv(
    csv_path: str,
) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    """
    Load an arbitrary rephrase CSV with the same schema as the Cochrane files.

    Expected columns: 'complex', 'simple' (string repr of a list), and optionally
    'label' and 'pair_id'. Intended for external augmentation data produced by
    experiments/sentence_level/build_external_rephrase.py.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Tuple of (complex_sentences, simple_references, labels, pair_ids), matching
        the output of load_cochrane_sentences().
    """
    print(f"Loading rephrase data from {csv_path}...")
    df = pd.read_csv(csv_path)

    complex_sentences = []
    simple_references = []
    labels = []
    pair_ids = []

    for idx, row in df.iterrows():
        complex_sent = row["complex"]
        simple_list = parse_simple_column(row["simple"])
        label = row["label"] if "label" in df.columns else "rephrase"
        pair_id = row["pair_id"] if "pair_id" in df.columns else f"row_{idx}"

        if pd.isna(complex_sent) or str(complex_sent).strip() == "":
            continue

        complex_sentences.append(str(complex_sent).strip())
        simple_references.append(simple_list)
        labels.append(label)
        pair_ids.append(pair_id)

    print(f"Loaded {len(complex_sentences)} rephrase pairs from {csv_path}")
    return complex_sentences, simple_references, labels, pair_ids


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    print("\n" + "="*80)
    print("Loading test set...")
    complex, simple, labels, ids = load_cochrane_sentences(split="test")
    
    print(f"\nFirst example:")
    print(f"Complex: {complex[0]}")
    print(f"Simple: {simple[0]}")
    print(f"Label: {labels[0]}")
    print(f"ID: {ids[0]}")
