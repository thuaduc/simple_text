"""Data loader for Cochrane sentence-level simplification data."""

import ast
import pandas as pd
from typing import List, Dict, Tuple


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
    data_dir: str = "cochrane/data"
) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    """
    Load Cochrane sentence-level simplification data.
    
    Args:
        split: One of 'train', 'val', or 'test'
        data_dir: Path to data directory containing CSV files
    
    Returns:
        Tuple of (complex_sentences, simple_references, labels, pair_ids)
        - complex_sentences: List of complex input sentences
        - simple_references: List of lists (multiple references per input)
        - labels: List of simplification operation labels
        - pair_ids: List of document IDs
    """
    csv_path = f"{data_dir}/cochraneauto_sents_{split}.csv"
    
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


def get_few_shot_examples(
    num_shots: int = 3,
    data_dir: str = "cochrane/data",
    seed: int = 42
) -> List[Dict[str, str]]:
    """
    Select diverse few-shot examples from training data.
    
    Selects examples covering different simplification operations:
    - Rephrase (paraphrase)
    - Split (one to many sentences)
    - Delete (remove information)
    
    Args:
        num_shots: Number of examples to select
        data_dir: Path to data directory
        seed: Random seed for reproducibility
    
    Returns:
        List of example dictionaries with 'complex' and 'simple' keys
    """
    complex_sents, simple_refs, labels, _ = load_cochrane_sentences(
        split="train",
        data_dir=data_dir
    )
    
    df = pd.DataFrame({
        'complex': complex_sents,
        'simple': simple_refs,
        'label': labels
    })
    
    # Filter for good examples (has non-empty simple reference)
    df = df[df['simple'].apply(lambda x: len(x) > 0)]
    
    examples = []
    
    # Try to get diverse examples by operation type
    target_labels = ['rephrase', 'split', 'delete']
    
    for label in target_labels:
        if len(examples) >= num_shots:
            break
        
        label_df = df[df['label'] == label]
        if len(label_df) > 0:
            # Sample one example of this type
            sample = label_df.sample(n=1, random_state=seed + len(examples))
            
            complex = sample.iloc[0]['complex']
            simple_list = sample.iloc[0]['simple']
            
            # For split operations, join multiple sentences
            # For others, use the first reference
            if label == 'split' and len(simple_list) > 1:
                simple = ' '.join(simple_list)
            else:
                simple = simple_list[0] if simple_list else ""
            
            if simple:  # Only add if we have a simple version
                examples.append({
                    'complex': complex,
                    'simple': simple
                })
    
    # If we still need more examples, sample randomly from rephrase
    while len(examples) < num_shots:
        remaining = num_shots - len(examples)
        rephrase_df = df[df['label'] == 'rephrase']
        
        if len(rephrase_df) == 0:
            break
        
        samples = rephrase_df.sample(
            n=min(remaining, len(rephrase_df)),
            random_state=seed + 100 + len(examples)
        )
        
        for _, row in samples.iterrows():
            simple_list = row['simple']
            if simple_list:
                examples.append({
                    'complex': row['complex'],
                    'simple': simple_list[0]
                })
    
    print(f"Selected {len(examples)} few-shot examples")
    return examples[:num_shots]


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
    
    print("\n" + "="*80)
    print("Getting few-shot examples...")
    examples = get_few_shot_examples(num_shots=3)
    
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Complex: {ex['complex'][:100]}...")
        print(f"Simple: {ex['simple'][:100]}...")
